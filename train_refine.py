"""
Train MattingRefine

Supports multi-GPU training with DistributedDataParallel() and SyncBatchNorm.
Select GPUs through CUDA_VISIBLE_DEVICES environment variable.

Example:

    CUDA_VISIBLE_DEVICES=0,1 python3 train_refine.py \
        --dataset-name videomatte240k \
        --model-backbone resnet50 \
        --model-name with-pedestrian-refine-2\
        --model-last-checkpoint "/eva_data/kie/research/BGMwd/checkpoint/with-pedestrian-refine/epoch-1-iter-63999.pth" \
        --epoch-end 4\
        --batch-size 4\
        --log-train-images-interval 200

"""

import argparse
import kornia
import torch
import os
import random

from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image

from V2wd.data_path import DATA_PATH
from V2wd.dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from V2wd.dataset import augmentation as A
from V2wd.model import MattingRefine
from V2wd import loss as LOSS
from V2wd.model.utils import load_matched_state_dict
from depth_estimator import Midas_depth, Normalize
import numpy as np


# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str,
                    required=True, choices=DATA_PATH.keys())

parser.add_argument('--model-backbone', type=str, required=True,
                    choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str,
                    default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-thresholding', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size',
                    type=int, default=3, choices=[1, 3])
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-last-checkpoint', type=str, default=None)

parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

parser.add_argument('--log-train-loss-interval', type=int, default=10)
parser.add_argument('--log-train-images-interval', type=int, default=1000)
parser.add_argument('--log-valid-interval', type=int, default=2000)

parser.add_argument('--checkpoint-interval', type=int, default=2000)

args = parser.parse_args()


distributed_num_gpus = torch.cuda.device_count()
assert args.batch_size % distributed_num_gpus == 0


# --------------- Main ---------------

MD = None


def train_worker(rank, addr, port):
    global MD
    MD = Midas_depth(device="cuda:{}".format(rank))
    # Distributed Setup
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=distributed_num_gpus)

    # Training DataLoader
    dataset_train = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['fgr'], mode='RGB'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize(
                (2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            A.PairRandomBoxBlur(0.1, 5),
            A.PairRandomSharpen(0.1),
            A.PairApplyOnlyAtIndices(
                [1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['fgr'], mode='RGB'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize(
                (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.1, 0.5), shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            A.PairApplyOnlyAtIndices(
                [1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True, shuffle=True),
        ImagesDataset(DATA_PATH['backgrounds']['train'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize(
                (2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 2), shear=(-5, 5)),
            T.RandomHorizontalFlip(),
            A.RandomBoxBlur(0.1, 5),
            A.RandomSharpen(0.1),
            T.ColorJitter(0.15, 0.15, 0.15, 0.05),
            T.ToTensor()
        ])),
    ])
    dataset_train_len_per_gpu_worker = int(
        len(dataset_train) / distributed_num_gpus)
    dataset_train = Subset(dataset_train, range(
        rank * dataset_train_len_per_gpu_worker, (rank + 1) * dataset_train_len_per_gpu_worker))
    dataloader_train = DataLoader(dataset_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  batch_size=args.batch_size // distributed_num_gpus,
                                  num_workers=args.num_workers // distributed_num_gpus)

    # Validation DataLoader
    if rank == 0:
        dataset_valid = ZipDataset([
            ZipDataset([
                ImagesDataset(DATA_PATH[args.dataset_name]
                              ['valid']['pha'], mode='L'),
                ImagesDataset(DATA_PATH[args.dataset_name]
                              ['valid']['fgr'], mode='RGB')
            ], transforms=A.PairCompose([
                A.PairRandomAffineAndResize(
                    (2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
                A.PairApply(T.ToTensor())
            ]), assert_equal_length=True),
            ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
                A.RandomAffineAndResize(
                    (2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
                T.ToTensor()
            ])),
        ])
        dataset_valid = SampleDataset(dataset_valid, 50)
        dataloader_valid = DataLoader(dataset_valid,
                                      pin_memory=True,
                                      drop_last=True,
                                      batch_size=args.batch_size // distributed_num_gpus,
                                      num_workers=args.num_workers // distributed_num_gpus)

    # Model
    model = MattingRefine(args.model_backbone,
                          args.model_backbone_scale,
                          args.model_refine_mode,
                          args.model_refine_sample_pixels,
                          args.model_refine_thresholding,
                          args.model_refine_kernel_size).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_distributed = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank])

    if args.model_last_checkpoint is not None:
        load_matched_state_dict(model, torch.load(args.model_last_checkpoint))

    optimizer = Adam([
        {'params': model.backbone.parameters(), 'lr': 5e-5},
        {'params': model.aspp.parameters(), 'lr': 5e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': model.refiner.parameters(), 'lr': 3e-4},
    ])
    scaler = GradScaler()

    # Logging and checkpoints
    if rank == 0:
        if not os.path.exists(f'checkpoint/{args.model_name}'):
            os.makedirs(f'checkpoint/{args.model_name}')
        writer = SummaryWriter(f'log/{args.model_name}')

    # Run loop
    for epoch in range(args.epoch_start, args.epoch_end):
        for i, ((true_pha, true_fgr), (p_mask, pedestrian), true_bgr) in enumerate(tqdm(dataloader_train)):
            step = epoch * len(dataloader_train) + i

            true_pha = true_pha.to(rank, non_blocking=True)
            true_fgr = true_fgr.to(rank, non_blocking=True)
            true_bgr = true_bgr.to(rank, non_blocking=True)
            pedestrian = pedestrian.to(rank, non_blocking=True)
            p_mask = p_mask.to(rank, non_blocking=True)
            true_pha, true_fgr, true_bgr, pedestrian, p_mask = random_crop(
                true_pha, true_fgr, true_bgr, pedestrian, p_mask)

            true_src = true_bgr.clone()

            # Augment with shadow
            aug_shadow_idx = torch.rand(len(true_src)) < 0.3
            if aug_shadow_idx.any():
                aug_shadow = true_pha[aug_shadow_idx].mul(
                    0.3 * random.random())
                aug_shadow = T.RandomAffine(
                    degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=(-5, 5))(aug_shadow)
                aug_shadow = kornia.filters.box_blur(
                    aug_shadow, (random.choice(range(20, 40)),) * 2)
                true_src[aug_shadow_idx] = true_src[aug_shadow_idx].sub_(
                    aug_shadow).clamp_(0, 1)
                del aug_shadow
            del aug_shadow_idx

            # Add pedestrain onto background
            # pedestrian = kornia.center_crop(pedestrian, (255,255))
            
            true_src = pedestrian * p_mask + (1 - p_mask) * true_src


            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)

            # Augment with noise
            aug_noise_idx = torch.rand(len(true_src)) < 0.4
            if aug_noise_idx.any():
                true_src[aug_noise_idx] = true_src[aug_noise_idx].add_(torch.randn_like(
                    true_src[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
                true_bgr[aug_noise_idx] = true_bgr[aug_noise_idx].add_(torch.randn_like(
                    true_bgr[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            del aug_noise_idx

            # feed into MiDas for depth estimate

            depth_input = true_src.cpu().numpy()
            depth_input = np.moveaxis(depth_input, 1, -1)
            true_depth = MD.inference(depth_input)
            true_depth = F.interpolate(true_depth,
                                       scale_factor=args.model_backbone_scale,
                                       mode='bilinear',
                                       align_corners=False,
                                       recompute_scale_factor=True)

            # Augment background with jitter
            aug_jitter_idx = torch.rand(len(true_src)) < 0.8
            if aug_jitter_idx.any():
                true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(
                    0.18, 0.18, 0.18, 0.1)(true_bgr[aug_jitter_idx])
            del aug_jitter_idx

            # Augment background with affine
            aug_affine_idx = torch.rand(len(true_bgr)) < 0.3
            if aug_affine_idx.any():
                true_bgr[aug_affine_idx] = T.RandomAffine(
                    degrees=(-1, 1), translate=(0.01, 0.01))(true_bgr[aug_affine_idx])
            del aug_affine_idx

            with autocast():
                pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _, pred_depth_sm = model_distributed(
                    true_src, true_bgr)
                matting_loss = LOSS.compute_mattingrefine_loss(
                    pred_pha, pred_fgr, pred_pha_sm,
                    pred_fgr_sm, pred_err_sm, true_pha, true_fgr)

                depth_loss = LOSS.compute_depth_loss(pred_depth_sm, true_depth)
                loss = matting_loss + depth_loss / 30

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0:
                if (i + 1) % args.log_train_loss_interval == 0:
                    writer.add_scalar('loss', loss, step)
                    writer.add_scalar('matting loss', matting_loss, step)
                    writer.add_scalar('depth loss', depth_loss, step)

                if (i + 1) % args.log_train_images_interval == 0:
                    writer.add_image('train_pred_pha',
                                     make_grid(pred_pha, nrow=5), step)
                    writer.add_image('train_pred_fgr',
                                     make_grid(pred_fgr, nrow=5), step)
                    writer.add_image('train_pred_com', make_grid(
                        pred_fgr * pred_pha, nrow=5), step)
                    writer.add_image('train_pred_err',
                                     make_grid((255 * pred_err_sm).to(torch.uint8), nrow=5), step)
                    writer.add_image('train_pred_depth',
                                     make_grid((255 * Normalize(pred_depth_sm)).to(torch.uint8), nrow=5), step)
                    writer.add_image('train_true_src',
                                     make_grid(true_src, nrow=5), step)
                    writer.add_image('train_true_bgr',
                                     make_grid(true_bgr, nrow=5), step)

                    writer.add_image('train_true_depth',
                                     make_grid((255 * Normalize(true_depth)).to(torch.uint8), nrow=5), step)

                del true_pha, true_fgr, true_src, true_bgr
                del pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm

                if (i + 1) % args.log_valid_interval == 0:
                    valid(model, dataloader_valid, writer, step)

                if (step + 1) % args.checkpoint_interval == 0:
                    torch.save(model.state_dict(
                    ), f'checkpoint/{args.model_name}/epoch-{epoch}-iter-{step}.pth')

        if rank == 0:
            torch.save(model.state_dict(),
                       f'checkpoint/{args.model_name}/epoch-{epoch}.pth')

    # Clean up
    dist.destroy_process_group()


# --------------- Utils ---------------


def random_crop(*imgs):
    H_src, W_src = imgs[0].shape[2:]
    W_tgt = random.choice(range(1024, 2048)) // 4 * 4
    H_tgt = random.choice(range(1024, 2048)) // 4 * 4
    scale = max(W_tgt / W_src, H_tgt / H_src)
    results = []
    for img in imgs:
        img = kornia.resize(img, (int(H_src * scale), int(W_src * scale)))
        img = kornia.center_crop(img, (H_tgt, W_tgt))
        results.append(img)
    return results


def valid(model, dataloader, writer, step):
    model.eval()
    loss_total = 0
    loss_count = 0
    with torch.no_grad():
        for (true_pha, true_fgr), true_bgr in dataloader:
            batch_size = true_pha.size(0)

            true_pha = true_pha.cuda(non_blocking=True)
            true_fgr = true_fgr.cuda(non_blocking=True)
            true_bgr = true_bgr.cuda(non_blocking=True)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            depth_input = true_src.cpu().numpy()
            depth_input = np.moveaxis(depth_input, 1, -1)
            true_depth = MD.inference(depth_input)
            true_depth = F.interpolate(true_depth,
                                       scale_factor=args.model_backbone_scale,
                                       mode='bilinear',
                                       align_corners=False,
                                       recompute_scale_factor=True)
            pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _, pred_depth_sm = model(
                true_src, true_bgr)
            loss = LOSS.compute_mattingrefine_loss(
                pred_pha, pred_fgr,
                pred_pha_sm, pred_fgr_sm,
                pred_err_sm, true_pha, true_fgr) + \
                LOSS.compute_depth_loss(pred_depth_sm, true_depth) / 30
            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

    writer.add_scalar('valid_loss', loss_total / loss_count, step)
    model.train()


# --------------- Start ---------------


if __name__ == '__main__':
    addr = 'localhost'
    port = str(random.choice(range(12300, 12400)))  # pick a random port.
    mp.spawn(train_worker,
             nprocs=distributed_num_gpus,
             args=(addr, port),
             join=True)

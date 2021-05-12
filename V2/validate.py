"""
Inference images: Extract matting on images.

Example:

    CUDA_VISIBLE_DEVICES=0 python V2/validate.py \
        --dataset-name "photomatte85" \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --model-checkpoint "/eva_data/kie/research/BGMwd/checkpoint/mattingrefine/epoch-7.pth" \


"""

import argparse
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from torch.nn import functional as F

from dataset import ImagesDataset, ZipDataset
from data_path import DATA_PATH
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference images')

parser.add_argument('--dataset-name', type=str,
                    required=True, choices=DATA_PATH.keys())
parser.add_argument('--model-type', type=str, required=True,
                    choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True,
                    choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str,
                    default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--device', type=str,
                    choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('-y', action='store_true')

args = parser.parse_args()


# --------------- Main ---------------


device = torch.device(args.device)

# Load model
if args.model_type == 'mattingbase':
    model = MattingBase(args.model_backbone)
if args.model_type == 'mattingrefine':
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)

model = model.to(device).eval()
model.load_state_dict(torch.load(args.model_checkpoint,
                      map_location=device), strict=False)


# Validation DataLoader
dataset_valid = ZipDataset([
    ZipDataset([
        ImagesDataset(DATA_PATH[args.dataset_name]['valid']['pha'], mode='L'),
        ImagesDataset(DATA_PATH[args.dataset_name]['valid']['fgr'], mode='RGB')
    ], transforms=A.PairCompose([
        A.PairRandomAffineAndResize(
            (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
        A.PairApply(T.ToTensor())
    ]), assert_equal_length=True),
    ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
        A.RandomAffineAndResize((512, 512), degrees=(-5, 5),
                                translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
        T.ToTensor()
    ])),
])

dataloader_valid = DataLoader(dataset_valid,
                              pin_memory=True,
                              batch_size=1,
                              num_workers=args.num_workers)


# outputs 5 vals:
# alpha: MSE, SAD, GRAD, CONN, fgr: MSE
def compute_metric(pred_pha, pred_fgr, true_pha, true_fgr):
    a1 = F.mse_loss(pred_pha, true_pha)
    a2 = F.l1_loss(pred_pha, true_pha)
    a5 = F.mse_loss((pred_fgr * (true_pha > 0)), (true_fgr * (true_pha > 0)))
    return a1, a2, 0, 0, a5


def write_metric_file():
    pass


MSE_pha_loss = 0
MSE_fgr_loss = 0
SAD_loss = 0
GRAD_loss = 0
CONN_loss = 0

# Conversion loop
with torch.no_grad():
    for (true_pha, true_fgr), true_bgr in dataloader_valid:
        batch_size = true_pha.size(0)

        true_pha = true_pha.cuda(non_blocking=True)
        true_fgr = true_fgr.cuda(non_blocking=True)
        true_bgr = true_bgr.cuda(non_blocking=True)
        true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

        pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
        # compute metric
        mm = compute_metric(pred_pha, pred_fgr, true_pha, true_fgr)
        MSE_pha_loss += mm[0]
        SAD_loss += mm[1]
        MSE_fgr_loss += mm[4]

MSE_pha_loss /= len(dataset_valid)
SAD_loss /= len(dataloader_valid)
MSE_fgr_loss /= len(dataloader_valid)


print("MSE alpha:", float(MSE_pha_loss.cpu()))
print("SAD_loss:", float(SAD_loss.cpu()))
print("MSE fgr:", float(MSE_fgr_loss))
import torch
import kornia
from torch.nn import functional as F


def compute_depth_loss(pred_depth: torch.Tensor, true_depth: torch.Tensor):
    true_depth = true_depth.detach()
    depth_abs = (pred_depth-true_depth).abs() * (true_depth >= 0)
    c = depth_abs.max() / 5
    mask = depth_abs <= c
    depth_loss = ((mask * depth_abs).sum() +
                  ((~mask * depth_abs) ** 2 + c ** 2).sum() / (2*c)) / (mask.shape[2] * mask.shape[3])
    return depth_loss


def compute_mattingbase_loss(pred_pha: torch.Tensor, pred_fgr: torch.Tensor,
                             pred_err: torch.Tensor, true_pha: torch.Tensor, true_fgr: torch.Tensor):
    true_err = torch.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    matting_loss = F.l1_loss(pred_pha, true_pha) + \
        F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha)) + \
        F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk) + \
        F.mse_loss(pred_err, true_err)

    return matting_loss


def compute_mattingrefine_loss(pred_pha_lg: torch.Tensor, pred_fgr_lg: torch.Tensor, pred_pha_sm: torch.Tensor,
                               pred_fgr_sm: torch.Tensor, pred_err_sm: torch.Tensor, true_pha_lg: torch.Tensor, true_fgr_lg: torch.Tensor):
    true_pha_sm = kornia.resize(true_pha_lg, pred_pha_sm.shape[2:])
    true_fgr_sm = kornia.resize(true_fgr_lg, pred_fgr_sm.shape[2:])
    true_msk_lg = true_pha_lg != 0
    true_msk_sm = true_pha_sm != 0
    return F.l1_loss(pred_pha_lg, true_pha_lg) + \
        F.l1_loss(pred_pha_sm, true_pha_sm) + \
        F.l1_loss(kornia.sobel(pred_pha_lg), kornia.sobel(true_pha_lg)) + \
        F.l1_loss(kornia.sobel(pred_pha_sm), kornia.sobel(true_pha_sm)) + \
        F.l1_loss(pred_fgr_lg * true_msk_lg, true_fgr_lg * true_msk_lg) + \
        F.l1_loss(pred_fgr_sm * true_msk_sm, true_fgr_sm * true_msk_sm) + \
        F.mse_loss(kornia.resize(pred_err_sm, true_pha_lg.shape[2:]),
                   kornia.resize(pred_pha_sm, true_pha_lg.shape[2:]).sub(true_pha_lg).abs())

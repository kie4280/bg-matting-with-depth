import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
import pdb
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable


class alpha_loss(_Loss):
    def __init__(self, device: str = "gpu", dtype: torch.dtype = torch.float32) -> None:
        super(alpha_loss, self).__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, alpha, alpha_ground):
        L1 = nn.L1Loss()
        loss1 = L1(alpha, alpha_ground)
        sobel_kernel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=self.device, dtype=self.dtype)
        sobel_kernel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=self.device, dtype=self.dtype)
        sobel_kernel_x = sobel_kernel_x.reshape([1,1,3,3])
        sobel_kernel_y = sobel_kernel_y.reshape([1,1,3,3])

        edge_x_1 = F.conv2d(alpha, sobel_kernel_x)
        edge_y_1 = F.conv2d(alpha, sobel_kernel_y)
        edge_x_2 = F.conv2d(alpha_ground, sobel_kernel_x)
        edge_y_2 = F.conv2d(alpha_ground, sobel_kernel_y)
        loss2_x = L1(edge_x_1.squeeze_(), edge_x_2.squeeze_())
        loss2_y = L1(edge_y_1.squeeze_(), edge_y_2.squeeze_())
        loss2 = loss2_x + loss2_y

        return loss1 + loss2


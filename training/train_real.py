from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


import os
import time
import argparse
import numpy as np

from loss_functions import alpha_loss, foreground_loss, error_map_loss

#CUDA
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])


"""Parses arguments."""
parser = argparse.ArgumentParser(description='Training Background Matting on Adobe Dataset.')
parser.add_argument('-n', '--name', type=str, help='Name of tensorboard and model saving folders.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch Size.')
parser.add_argument('-res', '--reso', type=int, help='Input image resolution')
parser.add_argument('-init_model', '--init_model', type=str, help='Initial model file')

parser.add_argument('-epoch', '--epoch', type=int, default=15,help='Maximum Epoch')
parser.add_argument('-n_blocks1', '--n_blocks1', type=int, default=7,help='Number of residual blocks after Context Switching.')
parser.add_argument('-n_blocks2', '--n_blocks2', type=int, default=3,help='Number of residual blocks for Fg and alpha each.')

args=parser.parse_args()

if 

for i in range(args.epoch):
    pass




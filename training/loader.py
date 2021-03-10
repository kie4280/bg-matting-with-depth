
import os
from typing import Any
import torch
import pandas as pd
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pdb, random
from torch.utils.data import Dataset, DataLoader
import random, os, cv2

class VideoData(Dataset):
    """
    load video files as data
    """
    
    def __init__(self) -> None:
        super().__init__()
    def __len__(self):
        pass
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

def random_crop(img):
    pass

def random_translation(img):
    pass

def random_noise(img):
    pass

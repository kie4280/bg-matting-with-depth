import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import random


class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None, shuffle=False):
        self.transforms = transforms
        self.mode = mode
        files = [*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)]
        if shuffle:
            random.shuffle(files)
            self.filenames = files
        else:
            self.filenames = sorted(files)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transforms:
            img = self.transforms(img)

        return img

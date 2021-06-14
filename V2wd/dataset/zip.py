from torch.utils.data import Dataset
from typing import List
import numpy as np

class ZipDataset(Dataset):
    def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False, shuffle=False):
        self.datasets = datasets
        self.transforms = transforms
        self.mapping = np.arange(len(datasets[0]))
        self._can_shuffle = shuffle and assert_equal_length
        
        if assert_equal_length:
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
        if self._can_shuffle:
            np.random.shuffle(self.mapping)
                

    def __len__(self):
        return max(len(d) for d in self.datasets)
    
    def __getitem__(self, idx):
        if self._can_shuffle:
            x = tuple(d[self.mapping[idx % len(d)]] for d in self.datasets)
        else:
            x = tuple(d[idx % len(d)] for d in self.datasets)
        if self.transforms:
            x = self.transforms(*x)
        return x

import torch
from torch.utils.data import Dataset
import numpy as np


class SpectralDataset(Dataset):
    def __init__(self, source=None, x=None, y=None, intermediate=None):
        if x is not None and y is not None:
            y = np.expand_dims(y, axis=1)
            source = np.concatenate((x,y), axis=1)

        self.df = source
        self.x = source[:,0:-1]
        self.y = source[:,-1]
        self.intermediate = intermediate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = self.get_x()[idx]
        soc = self.get_y()[idx]
        if self.intermediate is None:
            return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)
        else:
            return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32), torch.tensor(self.intermediate[idx], dtype=torch.float32)

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def get_intermediate(self):
        return self.intermediate

    def get_si(self, sif):
        return sif(self.get_x())



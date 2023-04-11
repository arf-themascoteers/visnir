import torch
from torch.utils.data import Dataset
import numpy as np


class SpectralDataset(Dataset):
    def __init__(self, source=None, x=None, y=None, intermediate=None):
        if intermediate is None:
            intermediate = []
        self.intermediate = intermediate
        if x is not None and y is not None:
            y = np.expand_dims(y, axis=1)
            source = np.concatenate((x,y), axis=1)

        self.df = source
        self.x = source[:,0:-1]
        self.y = source[:,-1]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = torch.tensor(self.x[idx], dtype=torch.float32)
        soc = torch.tensor(self.y[idx], dtype=torch.float32)
        items = [this_x, soc]
        for item_index in self.intermediate:
            this_item = self.x[item_index]
            items.append(torch.tensor(this_item, dtype=torch.float32))
        return items

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def get_intermediate(self):
        return self.intermediate

    def get_si(self, sif):
        return sif(self.get_x())



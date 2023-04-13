import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, source, x=None, intermediate=None):
        self.NAN = -9999
        self.df = source
        if x is None:
            x = list(range(source.shape[1]-1))
        self.x = torch.tensor(source[:,x], dtype=torch.float32)

        if torch.isnan(self.x).sum() != 0:
            raise "NAN in X"
        if self.x[self.x == self.NAN].shape[0] != 0:
            raise "NAN in X"

        if intermediate is None:
            intermediate = []
        self.intermediate = torch.tensor(source[:,intermediate], dtype=torch.float32)
        self.y = torch.tensor(source[:,-1], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.intermediate[idx], self.y[idx]

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def get_intermediate(self):
        return self.intermediate




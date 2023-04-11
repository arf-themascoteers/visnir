import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, source, x=None, intermediate=None):
        self.df = source
        if x is None:
            x = list(range(source.shape[1]-1))
        self.x = source[:,x]

        if intermediate is None:
            intermediate = []
        self.intermediate = source[:,intermediate]
        self.y = source[:,-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = torch.tensor(self.x[idx], dtype=torch.float32)
        this_intermediate = torch.tensor(self.intermediate[idx], dtype=torch.float32)
        this_soc = torch.tensor(self.y[idx], dtype=torch.float32)
        return this_x, this_intermediate, this_soc

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def get_intermediate(self):
        return self.intermediate




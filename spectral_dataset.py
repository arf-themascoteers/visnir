import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, source, x):
        self.df = source
        if x is None:
            x = list(range(source.shape[1]-1))
        self.x = torch.tensor(source[:,x], dtype=torch.float32)
        self.y = torch.tensor(source[:,-1], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x




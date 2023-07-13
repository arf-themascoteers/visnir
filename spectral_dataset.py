import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, source):
        self.df = source
        self.x = torch.tensor(source[:,0:-2], dtype=torch.float32)
        self.y = torch.tensor(source[:,-2:], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx,0], self.y[idx,1]




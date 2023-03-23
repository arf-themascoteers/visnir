import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, source, nsoc = False):
        self.df = source
        self.x = source[:,0:3]
        self.y = source[:,-1]
        self.nsoc = nsoc
        if self.nsoc:
            self.nitrogen = source[:,-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = self.get_x()[idx]
        soc = self.get_y()[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def get_si(self, sif):
        return sif(self.get_x())



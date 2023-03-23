import torch
import torch.nn as nn


class ANNI(nn.Module):
    def __init__(self, size=3, mini=None):
        super().__init__()
        self.soc_vec = nn.Sequential(
            nn.Linear(size, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5)
        )
        self.mini = mini
        self.soc = nn.Sequential(
            nn.Linear(6,3),
            nn.LeakyReLU(),
            nn.Linear(3,1)
        )
        for param in self.mini.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.soc_vec(x)
        x2 = self.mini(x)
        x = torch.hstack((x1,x2))
        x = self.soc(x)
        return x


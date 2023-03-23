import torch
import torch.nn as nn


class PHH_ANN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.n = nn.Sequential(
            nn.Linear(size, 5),
            nn.LeakyReLU(),
            nn.Linear(5,1)
        )

    def forward(self, x):
        x = self.n(x)
        return x


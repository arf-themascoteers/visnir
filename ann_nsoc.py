import torch
import torch.nn as nn


class NSOC_ANN(nn.Module):
    def __init__(self, size=3, n_model=None):
        super().__init__()
        self.soc_vec = nn.Sequential(
            nn.Linear(size, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5)
        )
        self.n = n_model
        self.soc = nn.Sequential(
            nn.Linear(6,3),
            nn.LeakyReLU(),
            nn.Linear(3,1)
        )

    def forward(self, x):
        x1 = self.soc_vec(x)
        x2 = self.n(x)
        x = torch.hstack((x1,x2))
        x = self.soc(x)
        return x


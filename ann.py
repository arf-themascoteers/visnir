import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, 20),
            nn.LeakyReLU(),
            nn.Linear(20,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


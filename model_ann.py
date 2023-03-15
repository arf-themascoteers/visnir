import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


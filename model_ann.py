import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, size=3, mid=None):
        super().__init__()
        DEFAULT_MID_LAYERS = [30, 10]
        if mid is None:
            mid = DEFAULT_MID_LAYERS
        layer_size = [size] + mid + [1]
        self.fc = nn.Sequential()
        for i in range(len(layer_size) - 2):
            l1 = layer_size[i]
            l2 = layer_size[i+1]
            self.fc.append(nn.Linear(l1, l2))
            self.fc.append(nn.LeakyReLU())
        self.fc.append(nn.Linear(layer_size[-2], layer_size[-1]))

    def forward(self, x):
        x = self.fc(x)
        return x


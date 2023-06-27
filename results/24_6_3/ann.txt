import torch
import torch.nn as nn
from spectral_dataset import SpectralDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, device, train_ds, test_ds, alpha = 0.0):
        super().__init__()
        torch.manual_seed(1)
        self.device = device
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.alpha = alpha
        self.num_epochs = 600
        self.batch_size = 600
        self.lr = 0.01

        x_size = train_ds.get_x().shape[1]
        self.intermediate_size = train_ds.get_intermediate().shape[1]
        size = x_size

        if self.intermediate_size == 0:
            self.linear = nn.Sequential(
                nn.Linear(size, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 3),
                nn.LeakyReLU(),
                nn.Linear(3, 1)
            )
        else:
            self.soc_vec = nn.Sequential(
                nn.Linear(size, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 4)
            )

            self.inter_vec = nn.Sequential(
                nn.Linear(size,10),
                nn.LeakyReLU(),
                nn.Linear(10,self.intermediate_size)
            )
            self.soc = nn.Sequential(
                nn.Linear(4 + self.intermediate_size,4),
                nn.LeakyReLU(),
                nn.Linear(4,1)
            )

    def forward(self, x):
        if self.intermediate_size == 0:
            x = self.linear(x)
            return x, None
        x1 = self.soc_vec(x)
        x2 = self.inter_vec(x)
        x = torch.hstack((x1,x2))
        x = F.leaky_relu(x)
        x = self.soc(x)
        return x, x2

    def train_model(self):
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')
        n_batches = int(len(self.train_ds)/self.batch_size) + 1

        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            batch_number = 0
            for (x, intermediate, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat, intermediate_hat = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                if intermediate.shape[1] !=0:
                    intermediate = intermediate.to(self.device)
                    loss_intermediate = criterion(intermediate_hat, intermediate)
                    loss = (1-self.alpha) * loss + self.alpha * loss_intermediate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    def test(self):
        batch_size = 30000
        self.eval()
        self.to(self.device)

        dataloader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)

        for (x, intermediate, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat, intermediate = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            return r2

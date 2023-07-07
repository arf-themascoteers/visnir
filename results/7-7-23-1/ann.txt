import torch
import torch.nn as nn
from spectral_dataset import SpectralDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, device, train_ds, test_ds, config="rgb",alpha = 0.0):
        super().__init__()
        torch.manual_seed(1)
        self.device = device
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.alpha = alpha
        self.num_epochs = 600
        self.batch_size = 600
        self.lr = 0.01
        self.linear = nn.Sequential(
            nn.Linear(3, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 3),
            nn.LeakyReLU(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

    def train_model(self):
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')
        n_batches = int(len(self.train_ds)/self.batch_size) + 1

        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            batch_number = 0
            for (x, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                n = y[:,0]
                oc = y[:,1]
                y_hat = self(x)
                n_hat = y_hat[:,0].reshape(-1)
                oc_hat = y_hat[:,1].reshape(-1)
                loss_n = criterion(n_hat, n)
                loss_oc = criterion(oc_hat, oc)
                loss_phy_1 = torch.sum(F.relu( n - oc_hat))
                loss_phy_2 = torch.sum(F.relu( n_hat - oc))
                loss = loss_n + loss_oc + (self.alpha * loss_phy_1) + (self.alpha * loss_phy_2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                # print(f'Epoch:{epoch + 1} (of {self.num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')
                # print(f'L_P1:{self.alpha*loss_phy_1:0.3f}\tL_P2:{self.alpha*loss_phy_2:0.3f}'
                #       f'\tL_N:{loss_n:0.3f}\tL_OC:{loss_oc:0.3f}')

    def test(self):
        batch_size = 30000
        self.eval()
        self.to(self.device)

        dataloader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            n = y[:, 0]
            oc = y[:, 1]
            y_hat = self(x)
            n_hat = y_hat[:, 0].reshape(-1)
            oc_hat = y_hat[:, 1].reshape(-1)
            n = n.detach().cpu().numpy()
            oc = oc.detach().cpu().numpy()
            n_hat = n_hat.detach().cpu().numpy()
            oc_hat = oc_hat.detach().cpu().numpy()
            r2_n = r2_score(n, n_hat)
            r2_oc = r2_score(oc, oc_hat)
            return r2_n, r2_oc

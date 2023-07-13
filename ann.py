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
        self.TEST = False
        self.device = device
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.alpha = alpha
        self.num_epochs = 300
        self.batch_size = 600
        self.lr = 0.01

        x_size = train_ds.get_x().shape[1]
        self.intermediate_size = train_ds.get_intermediate().shape[1]
        size = x_size

        self.common = nn.Sequential(
            nn.Linear(size, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 5)
        )

        self.n = nn.Sequential(
            nn.Linear(5,3),
            nn.LeakyReLU(),
            nn.Linear(3, 1)
        )

        self.oc = nn.Sequential(
            nn.Linear(5,3),
            nn.LeakyReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        common = self.common(x)
        n = self.n(common)
        oc = self.oc(common)
        return n, oc

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')
        n_batches = int(len(self.train_ds)/self.batch_size) + 1

        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            batch_number = 0
            for (x, n, oc) in dataloader:
                x = x.to(self.device)
                n = n.to(self.device)
                oc = oc.to(self.device)
                n_hat, oc_hat = self(x)
                n_hat = n_hat.reshape(-1)
                oc_hat = oc_hat.reshape(-1)
                loss_n = criterion(n_hat, n)
                loss_oc = criterion(oc_hat, oc)
                loss_phy = torch.sum(F.relu(n_hat-oc_hat)) # add all combination # add reduced combinations
                loss = loss_n + loss_oc + (self.alpha * loss_phy)
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

        for (x, n, oc) in dataloader:
            x = x.to(self.device)
            n = n.to(self.device)
            oc = oc.to(self.device)
            n_hat, oc_hat = self(x)
            n_hat = n_hat.reshape(-1)
            oc_hat = oc_hat.reshape(-1)

            n = n.detach().cpu().numpy()
            n_hat = n_hat.detach().cpu().numpy()
            r2_n = r2_score(n, n_hat)

            oc = oc.detach().cpu().numpy()
            oc_hat = oc_hat.detach().cpu().numpy()
            r2_oc = r2_score(oc, oc_hat)

            return r2_n, r2_oc

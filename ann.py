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
        self.loss_function = "normal"
        if config == "rgbnp":
            self.loss_function = "rgbnp"

        size = train_ds.get_x().shape[1]

        self.linear = nn.Sequential(
            nn.Linear(size, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 3),
            nn.LeakyReLU(),
            nn.Linear(3, 1)
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
                y_hat = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                if self.loss_function == "rgbnp":
                    if epoch > 500:
                        nitrogen = x[:,-1].reshape(-1)
                        loss_phy = torch.sum(F.relu( nitrogen - y_hat))
                        loss = (1-self.alpha) * loss + (self.alpha * loss_phy)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                #print(f'Epoch:{epoch + 1} (of {self.num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    def test(self):
        batch_size = 30000
        self.eval()
        self.to(self.device)

        dataloader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            return r2

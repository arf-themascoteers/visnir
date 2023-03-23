import torch
from spectral_dataset import SpectralDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time

def test(device, ds, model, return_pred = False, shuffle=False):
    batch_size = 30000
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    model.eval()
    model.to(device)

    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        r2 = r2_score(y, y_hat)
        if return_pred:
            return r2, y_hat
        return r2


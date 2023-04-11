from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from spectral_dataset import SpectralDataset
import numpy as np


def test(device, ds, model, machine=None, return_pred = False, shuffle=False):
    batch_size = 30000
    if machine == "annx":
        y = ds.y
        x = ds.x[:,0:-1]
        ds = SpectralDataset(x=x, y=y, intermediate=np.zeros_like(y))
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    model.eval()
    model.to(device)

    if machine == "ann":
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
    else:#annx
        for (x, y, intermediate) in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat, intermediate = model(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            if return_pred:
                return r2, y_hat
            return r2



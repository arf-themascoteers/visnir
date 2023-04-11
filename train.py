import torch
from torch.utils.data import DataLoader
from ann import ANN
from annx import ANNX
from spectral_dataset import SpectralDataset


def train(device, ds:SpectralDataset, machine="ann"):
    torch.manual_seed(1)
    num_epochs = 200
    batch_size = 600
    lr = 0.001
    x_size = ds.get_x().shape[1]
    if machine == "ann":
        model = ANN(size=x_size)
    elif machine == "annx":
        model = ANNX(size=x_size - 1)
        y = ds.y
        x = ds.x[:,0:-1]
        intermediate = ds.x[:,-1]
        ds = SpectralDataset(x=x, y=y, intermediate=intermediate)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = torch.nn.MSELoss(reduction='sum')
    n_batches = int(len(ds)/batch_size) + 1
    alpha = 0.3
    for epoch in range(num_epochs):
        batch_number = 0
        if machine == "ann":
            for (x, y) in dataloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
        else:#annx
            for (x, y, intermediate) in dataloader:
                x = x.to(device)
                y = y.to(device)
                intermediate = intermediate.to(device)
                y_hat, intermediate_hat = model(x)
                y_hat = y_hat.reshape(-1)
                intermediate_hat = intermediate_hat.reshape(-1)
                loss_y = criterion(y_hat, y)
                loss_intermediate = criterion(intermediate_hat, intermediate)
                loss = loss_y + alpha * loss_intermediate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1

        #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    return model

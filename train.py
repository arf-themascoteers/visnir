import torch
from torch.utils.data import DataLoader
from ann import ANN
from anni import ANNI
from annmini import ANN_Mini
from annl1 import ANN_L1
from annl2 import ANN_L2
from spectral_dataset import SpectralDataset


def train(device, ds:SpectralDataset, machine="ann"):
    torch.manual_seed(0)
    num_epochs = 200
    batch_size = 600
    lr = 0.001
    x_size = ds.get_x().shape[1]
    if machine == "ann":
        model = ANN(size=x_size)
    elif machine == "annmini":
        model = ANN_Mini(size=x_size-1)
        y = ds.x[:,-1]
        x = ds.x[:,0:-1]
        ds = SpectralDataset(x=x, y=y)
    elif machine == "anni":
        minimodel = train(device, ds, machine="annmini")
        model = ANNI(size=x_size-1, mini=minimodel)
        y = ds.y
        x = ds.x[:,0:-1]
        ds = SpectralDataset(x=x, y=y)
    elif machine == "annl1":
        model = ANN_L1(size=x_size-1)
        y = ds.y
        x = ds.x[:,0:-1]
        intermediate = ds.x[:,-1]
        ds = SpectralDataset(x=x, y=y, intermediate=intermediate)
    elif machine == "annl2":
        model = ANN_L2(size=x_size-1)
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
        if machine == "annl1":
            for (x, y, intermediate) in dataloader:
                x = x.to(device)
                y = y.to(device)
                intermediate = intermediate.to(device)
                y_hat, intermediate_hat = model(x)
                y_hat = y_hat.reshape(-1)
                loss_y = criterion(y_hat, y)
                loss_intermediate = criterion(intermediate_hat, intermediate)
                loss = loss_y + alpha * loss_intermediate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
        if machine == "annl2":
            for (x, y, intermediate) in dataloader:
                x = x.to(device)
                y = y.to(device)
                intermediate = intermediate.to(device)
                y_hat = model(x)
                y_hat = y_hat.reshape(-1)
                loss_y = criterion(y_hat, y)
                loss_y.backward()
                optimizer.step()
                optimizer.zero_grad()

                intermediate_hat = model.n_only(x)
                loss_intermediate = criterion(intermediate_hat, intermediate)
                loss_intermediate.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_number += 1
        else:
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
        #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    return model

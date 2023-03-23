import ds_manager
import torch
from train import train
from test import test
from ann import ANN
from n_ann import PHH_ANN
from ann_nsoc import NSOC_ANN


n_dset = ds_manager.DSManager("lucas", "phh")
train_ds = n_dset.train_ds
test_ds = n_dset.test_ds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_model = train(device, train_ds)

torch.save(n_model,"n_model.h5")

result = test(device, test_ds, n_model)
print(result)
n_model.train()

# for param in n_model.parameters():
#     param.requires_grad = False

nsoc = NSOC_ANN(n_model=n_model)
dset = ds_manager.DSManager("lucas", "oc", nsoc=True)
train_ds = dset.train_ds
test_ds = dset.test_ds
n_model = train(device, train_ds, model=nsoc)
torch.save(n_model, "nsoc.h5")

result = test(device, test_ds, n_model)
print(result)

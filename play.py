import torch
import torch.nn.functional as F

x1 = torch.tensor([3,3,3,3,3,3], dtype=torch.float32)
x2 = torch.tensor([1,1,1,1,1,1], dtype=torch.float32)

z = F.mse_loss(x1,x2)

print(z)
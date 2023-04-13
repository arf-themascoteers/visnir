from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype
from ds_manager import DSManager
import torch
import numpy as np

# dm = DSManager()
#
# for train, test in dm.get_k_folds():
#     print(train)
#     print(test)

# x = torch.tensor(list(range(100)))
#
# indices = torch.randperm(x.shape[0])[:10]
# y = x[indices]
#
# print(type(y))
# print(y)

x = np.array(list(range(100)))

indices = np.random.choice(x, size=90, replace=False)
y = x[indices]

print(type(y))
print(y)
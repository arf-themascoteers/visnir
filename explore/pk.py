import pandas as pd
import os
os.chdir("../")
df = pd.read_csv("data/vis.csv")
npdf = df.to_numpy()
print(npdf.shape)

x = 0
y = 0
for i in range(npdf.shape[0]):
    if npdf[i,7] > npdf[i,9]:
        x = x+1
    else:
        y = y+1

print(x)
print(y)
import numpy as np
import os
os.chdir("../")
import pandas as pd

df = pd.read_csv("data/vis_ells.csv")
lus = {}
for i in range(len(df)):
    row = df.iloc[i]
    lu_code = row["lu1"]
    if lu_code in lus:
        lus[lu_code] = lus[lu_code] + 1
    else:
        lus[lu_code] = 1

lus = {k: v for k, v in sorted(lus.items(), key=lambda item: item[1], reverse=True)}
for i,v in lus.items():
    print(f"{i} -> {v}")
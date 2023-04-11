import numpy as np
import os
os.chdir("../")
import pandas as pd

df = pd.read_csv("data/vis_ells.csv")
lcs = {}
for i in range(len(df)):
    row = df.iloc[i]
    lc_code = row["lc1"]
    if lc_code in lcs:
        lcs[lc_code] = lcs[lc_code] + 1
    else:
        lcs[lc_code] = 1

lcs = {k: v for k, v in sorted(lcs.items(), key=lambda item: item[1], reverse=True)}
for i,v in lcs.items():
    print(f"{i} -> {v}")
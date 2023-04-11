import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from pandas.api.types import is_numeric_dtype

df = pd.read_csv("data/vis_css.csv")
# y = pd.get_dummies(df.lc1,prefix="lc1")
# print(y)
# y = pd.get_dummies(df.lu1,prefix="lu1")
# print(y)

print(df.columns)
print(len(df.columns))

newdf = df.copy()

for col in df.columns:
    if not is_numeric_dtype(df[col]):
        newdf = newdf.drop(col, axis=1)
        y = pd.get_dummies(df[col],prefix=col)
        newdf = pd.concat([newdf, y], axis=1)

print(len(newdf.columns))
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from pandas.api.types import is_numeric_dtype

df = pd.read_csv("data/vis_css.csv")
print(len(df))
df2 = df[df.lc1 == "C21"]
print(len(df2))
# x = df.lc1.unique()
# print(x)
df = df.drop(df[df.lc1 == "C21"].index)
print(len(df))
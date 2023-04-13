import os
os.chdir("../")
import pandas as pd

NAN = -9999
vis = pd.read_csv("data/vis_with_empty.csv")

con = ((vis["clay"] == NAN) | (vis["sand"] == NAN) | (vis["silt"] == NAN))
with_empty = vis[con]
no_empty = vis[ ~con ]

with_empty.to_csv("data/with_empty.csv")
no_empty.to_csv("data/no_empty.csv")

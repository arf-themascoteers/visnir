import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    configs.append({"x":['lat','lon'], "y":"oc", "machine":"ann"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="latlon"
    )
    ev.process()
    print("Done all")
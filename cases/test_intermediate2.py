import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    configs.append({"x":["665", "560", "490"], "y":"oc", "machine":"ann"})
    configs.append({"x":["665", "560", "490"], "intermediate":["k"], "y":"oc", "machine":"annx"})
    configs.append({"x":["665", "560", "490"], "intermediate":["p"], "y":"oc", "machine":"annx"})
    configs.append({"x":["665", "560", "490"], "intermediate":["k","p"], "y":"oc", "machine":"annx"})

    ev = Evaluator(
        cofigs=configs,
        
        repeat=1,
        folds=10,
        prefix="tint2"
    )
    ev.process()
    print("Done all")
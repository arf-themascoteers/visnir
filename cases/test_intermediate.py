import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    configs.append({"x":["665", "560", "490"], "intermediate":["k","p"], "y":"oc", "machine":"annx"})

    ev = Evaluator(
        cofigs=configs,
        
        repeat=1,
        folds=10,
        prefix="tint"
    )
    ev.process()
    print("Done all")
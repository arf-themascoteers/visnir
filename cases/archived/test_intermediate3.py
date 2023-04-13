import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    configs.append({"x":["665", "560", "490"], "intermediate":["lc1","lu1"], "min_row":100, "y":"oc", "machine":"annx"})
    configs.append({"x":["665", "560", "490"]+["lc1"], "y":"oc", "min_row":100, "machine":"ann"})
    configs.append({"x":["665", "560", "490"]+["lc1","lu1"],"min_row":100, "y":"oc", "machine":"ann"})
    configs.append({"x":["665", "560", "490"]+["lc1"], "intermediate":["lu1"], "min_row":100, "y":"oc", "machine":"annx"})

    ev = Evaluator(
        cofigs=configs,

        repeat=1,
        folds=10,
        prefix="tint3"
    )
    ev.process()
    print("Done all")
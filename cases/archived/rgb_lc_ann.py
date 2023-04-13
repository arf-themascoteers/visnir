import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []
    for property in ["lc1"]:
        configs.append({"x":rgb, "y":"oc", "machine":"ann"})
        configs.append({"x":rgb+[property], "y":"oc", "machine":"ann"})
        configs.append({"x":[property], "y":"oc", "machine":"ann"})
        configs.append({"x":[property], "y":"oc", "machine":"ann","min_row":100})
        configs.append({"x":rgb+[property], "y":"oc", "machine":"ann","min_row":100})

    ev = Evaluator(
        cofigs=configs,

        repeat=1,
        folds=10,
        prefix="rgb_lc_ann"
    )
    ev.process()
    print("Done all")
import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []
    for property in ["lc1"]:
        configs.append({"x":rgb+[property], "y":"oc", "machine":"ann"})

    ev = Evaluator(
        cofigs=configs,
        file_name="vis_ells",
        repeat=1,
        folds=10,
        prefix="lc_ann"
    )
    ev.process()
    print("Done all")
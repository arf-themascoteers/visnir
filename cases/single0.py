import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = [{"x":rgb, "y":"oc",  "intermediate":["phc"]}]

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"single_0",
        files="data/vis_with_empty.csv",
        alpha=0
    )
    ev.process()

    print("Done all")
import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":

    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb + ["clay","sand","silt"], "y": "oc"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="rgb_css_only_small",
        files = "data/no_empty.csv"
    )
    ev.process()
    print("Done all")
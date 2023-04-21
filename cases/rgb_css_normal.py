import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc", "intermediate":["clay"]})
    configs.append({"x": ["clay"], "y": "oc"})
    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb + ["clay"], "y": "oc"})

    ev = Evaluator(
        cofigs=configs,
        repeat=5,
        folds=10,
        prefix="rgb_css_normal5",
        files = "data/no_empty.csv"
    )
    ev.process()
    print("Done all")
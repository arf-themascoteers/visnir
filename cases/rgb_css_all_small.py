import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb, "y": "oc", "intermediate":["clay"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["sand"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["silt"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["clay","sand"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["sand","silt"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["silt","clay"]})
    configs.append({"x": rgb, "y": "oc", "intermediate":["clay","sand","silt"]})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="rgb_css_all_small",
        files = "data/no_empty.csv"
    )
    ev.process()
    print("Done all")
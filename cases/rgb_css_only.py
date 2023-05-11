import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    train_file = "data/no_empty.csv"
    test_file = "data/with_empty.csv"

    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc", "intermediate":["clay","sand","silt"]})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="rgb_css_only4",
        files = (train_file, test_file),
        ratios=(1,0.06)
    )
    ev.process()
    print("Done all")
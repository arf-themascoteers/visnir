import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    train_file = "data/no_empty.csv"
    test_file = "data/with_empty.csv"

    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb+["clay"], "y": "oc"})
    configs.append({"x": rgb+["sand"], "y": "oc"})
    configs.append({"x": rgb+["silt"], "y": "oc"})
    configs.append({"x": rgb+["clay","sand"], "y": "oc"})
    configs.append({"x": rgb+["sand","silt"], "y": "oc"})
    configs.append({"x": rgb+["silt","clay"], "y": "oc"})
    configs.append({"x": rgb+["clay","sand","silt"], "y": "oc"})
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
        prefix="rgb_css_all_full",
        files = (train_file, test_file)
    )
    ev.process()
    print("Done all")
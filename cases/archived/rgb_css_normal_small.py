import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb + ["clay"], "y": "oc"})
    configs.append({"x": rgb + ["sand"], "y": "oc"})
    configs.append({"x": rgb + ["silt"], "y": "oc"})
    configs.append({"x": rgb + ["clay","sand"], "y": "oc"})
    configs.append({"x": rgb + ["sand","silt"], "y": "oc"})
    configs.append({"x": rgb + ["silt","clay"], "y": "oc"})
    configs.append({"x": rgb + ["clay","sand","silt"], "y": "oc"})

    ev = Evaluator(
        cofigs=configs,
        repeat=5,
        folds=10,
        prefix="rgb_css_normal5",
        files = "data/no_empty.csv"
    )
    ev.process()
    print("Done all")
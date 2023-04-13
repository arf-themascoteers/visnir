import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []
    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb + ["elevation"], "y": "oc"})


    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="elevation",
        files="data/with_rainfall.csv"
    )
    ev.process()
    print("Done all")
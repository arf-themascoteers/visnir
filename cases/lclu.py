import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []
    configs.append({"x": rgb, "y": "oc"})
    configs.append({"x": rgb + ["lc1"], "y": "oc"})
    configs.append({"x": rgb + ["lu1"], "y": "oc"})
    configs.append({"x": rgb + ["lc1","lu1"], "y": "oc"})
    configs.append({"x": rgb + ["lc1"], "y": "oc","min_row":100})
    configs.append({"x": rgb + ["lu1"], "y": "oc","min_row":100})
    configs.append({"x": rgb + ["lc1","lu1"], "y": "oc","min_row":100})


    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="lclu"
    )
    ev.process()
    print("Done all")
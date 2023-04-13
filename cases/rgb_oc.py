import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []
    configs.append({"x": rgb, "y": "oc", "machine": "ann"})

    ev = Evaluator(
        cofigs=configs,
        file_name="vis_css",
        repeat=1,
        folds=10,
        prefix="rgb_oc"
    )
    ev.process()
    print("Done all")
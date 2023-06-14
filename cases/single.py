import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []

    configs.append({"x": rgb, "y": "oc"})

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb + [property], "y":"oc"})

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb, "y":"oc",  "intermediate":[property]})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="single",
        files = "data/vis_with_empty.csv"
    )
    ev.process()
    print("Done all")
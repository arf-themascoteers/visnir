import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = [{"x":rgb, "y":"oc"}]
    for property in ["phc","phh","ec","caco3","p","n","k"]:
        configs.append({"x":rgb + [property], "y":"oc", "machine":"annl2"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="l2"
    )
    ev.process()
    print("Done all")
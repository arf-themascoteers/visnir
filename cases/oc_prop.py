import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":["oc"], "y":property})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"oc_prop"
    )
    ev.process()

    print("Done all")
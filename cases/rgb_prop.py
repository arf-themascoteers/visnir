import os
from evaluator import Evaluator


def process():
    rgb = ["665", "560", "490"]
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb, "y":property})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"rgb_prop"
    )
    ev.process()

    print("Done all")


if __name__ == "__main__":
    os.chdir("../")
    process()
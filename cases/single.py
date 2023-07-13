import os
from evaluator import Evaluator


def process():
    rgb = ["665", "560", "490"]
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb, "y":"oc",  "intermediate":[property]})

    alpha = 0
    while alpha <= 0.9:
        ev = Evaluator(
            cofigs=configs,
            repeat=1,
            folds=10,
            prefix=f"single_{alpha}",
            alpha=alpha
        )
        ev.process()
        alpha = alpha + 0.1

    print("Done all")


if __name__ == "__main__":
    os.chdir("../")
    process()
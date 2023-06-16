import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    alpha = 0.1
    rgb = ["665", "560", "490"]
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb, "y":"oc",  "intermediate":[property]})

    while alpha <= 0.7:
        ev = Evaluator(
            cofigs=configs,
            repeat=1,
            folds=10,
            prefix=f"single_{int(alpha*10)}",
            files="data/vis_with_empty.csv",
            alpha=alpha
        )
        ev.process()
        alpha = alpha + 0.1

    print("Done all")
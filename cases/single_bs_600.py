import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = []

    ev = Evaluator(
        cofigs=[{"x":rgb, "y":"oc"}],
        repeat=1,
        folds=10,
        prefix=f"oc",
        files="data/vis_with_empty.csv"
    )
    ev.process()



    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":rgb, "y":"oc",  "intermediate":[property]})

    alpha = 0
    while alpha <= 1:
        ev = Evaluator(
            cofigs=configs,
            repeat=1,
            folds=10,
            prefix=f"bs600_{int(alpha*10)}",
            files="data/vis_with_empty.csv",
            alpha=alpha
        )
        ev.process()
        alpha = alpha + 0.1

    print("Done all")
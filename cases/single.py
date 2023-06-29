import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = ["rgbnp","rgbn","rgb"]

    for alpha in range(10):
        alpha_val = alpha/10
        ev = Evaluator(
            cofigs=configs,
            repeat=1,
            folds=10,
            prefix=f"single_{alpha_val}",
            alpha=alpha_val
        )
        ev.process()

    print("Done all")
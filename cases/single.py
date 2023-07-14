import os
from evaluator import Evaluator


def process():
    for alpha in range(1,11):
        ev = Evaluator(
            repeat=1,
            folds=10,
            prefix=f"para_{alpha}",
            alpha=alpha/50
        )
        ev.process()

    print("Done all")


if __name__ == "__main__":
    os.chdir("../")
    process()
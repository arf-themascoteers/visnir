import os

os.chdir("../")
from evaluator import Evaluator


def process():
    rgb = ["665", "560", "490"]

    ev = Evaluator(
        cofigs=[{"x": rgb, "y": "oc"}],
        repeat=1,
        folds=10,
        prefix=f"oc"
    )
    ev.process()
    print("Done all")


if __name__ == "__main__":
    process()
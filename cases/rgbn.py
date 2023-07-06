import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    configs = ["rgbn"]
    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"rgbn"
    )
    ev.process()
    print("Done rgbn")
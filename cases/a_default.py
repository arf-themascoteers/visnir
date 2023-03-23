import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x":["665", "560", "490"], "y":"oc"},
            {"x":["665", "560", "490","n"], "y":"oc"},
            # {"x":["665", "560", "490"], "y":"n"},
            # {"x":["665", "560", "490","oc"], "y":"n"},
            # {"x":["n"], "y":"oc"},
            {"x":["oc"], "y":"n"}
        ],
        repeat=1,
        folds=2
    )
    ev.process()
    print("Done all")
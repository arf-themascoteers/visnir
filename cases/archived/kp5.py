import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x":["665", "560", "490"], "y":"oc"},
            {"x":["665", "560", "490","k"], "y":"oc"},
            {"x": ["k"], "y": "oc"},
            {"x": ["665", "560", "490"], "y": "k"},
        ],
        prefix="kp66"
    )
    ev.process()
    print("Done all")
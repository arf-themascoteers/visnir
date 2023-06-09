import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x": ["665", "560", "490"], "y": "k"},
            {"x": ["665", "560", "490"], "y": "p"},
            # {"x":["665", "560", "490"], "y":"oc"},
            # {"x":["665", "560", "490","k"], "y":"oc"},
            # {"x":["665", "560", "490","p"], "y":"oc"},
            # {"x":["665", "560", "490","k","p"], "y":"oc"},
            # {"x":["665", "560", "490","kp"], "y":"oc"},
            # {"x": ["k"], "y": "oc"},
            # {"x": ["p"], "y": "oc"},
            # {"x":["kp"], "y":"oc"},
            # {"x":["kp","k","p"], "y":"oc"},
            # {"x":["665", "560", "490","k","kp"], "y":"oc"},
            # {"x":["665", "560", "490","p","kp"], "y":"oc"},
            # {"x":["665", "560", "490","p","k","kp"], "y":"oc"},
            # {"x":["k"], "y":"p"},
            # {"x":["p"], "y":"k"},
        ],
        prefix="kp4"
    )
    ev.process()
    print("Done all")
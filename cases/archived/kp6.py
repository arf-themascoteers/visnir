import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x": ["665", "560", "490"], "y": "oc"},
            {"x": ["665", "560", "490", "k"], "y": "oc", "machine": "annx"},
            {"x": ["665", "560", "490", "k"], "y": "oc"},
            {"x": ["665", "560", "490"], "y": "k"},
            {"x": ["k"], "y": "oc"},
        ],
        prefix="kp66"
    )
    ev.process()
    print("Done all")
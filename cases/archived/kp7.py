import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x": ["665", "560", "490"], "y": "oc"},
            {"x": ["665", "560", "490", "k"], "y": "oc"},
            {"x": ["665", "560", "490", "k"], "y": "oc", "machine": "annx"}
        ],
        prefix="kp77",
        repeat=3,
        folds=10
    )
    ev.process()
    print("Done all")
import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        configs = [{"x":["665", "560", "490"], "y":"oc"}],
        prefix="j",
        verbose=True
    )
    ev.process()
    print("Done all")
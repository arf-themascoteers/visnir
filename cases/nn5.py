import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=["vis"],
        prefix="j",
        verbose=True
    )
    ev.process()
    print("Done all")
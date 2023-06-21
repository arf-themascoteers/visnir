import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":[property], "y":"oc"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"prop",
        files="data/vis_with_empty.csv",
        alpha=0
    )
    ev.process()

    print("Done all")
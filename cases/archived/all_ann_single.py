import os
os.chdir("../../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    for property in ["phc","phh","ec","caco3","p","n","k"]:
        configs.append({"x":[property], "y":"oc", "machine":"ann"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="all_ann_single"
    )
    ev.process()
    print("Done all")
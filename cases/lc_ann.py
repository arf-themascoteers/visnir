import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    for property in ["lc1"]:
        configs.append({"x":[property], "y":"oc", "machine":"ann", "file_name":"vis_ells"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="lc_ann"
    )
    ev.process()
    print("Done all")
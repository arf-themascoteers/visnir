import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    for property in ["lc1"]:
        configs.append({"x":[property], "y":"oc", "machine":"ann","min_row":100})
        configs.append({"x":[property], "y":"oc", "machine":"ann"})

    ev = Evaluator(
        cofigs=configs,
        file_name="vis_ells",
        repeat=1,
        folds=10,
        prefix="lc_ann"
    )
    ev.process()
    print("Done all")
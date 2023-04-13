import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = []
    for property in ["phc","phh","ec","caco3","p","n","k","elevation","lc1","lu1","stones","oc"]:
        configs.append({"x":[property], "y":"oc", "machine":"ann"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix="single",
        file_name="vis_ells"
    )
    ev.process()
    print("Done all")
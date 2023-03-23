import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    configs = [{"x":["665", "560", "490"], "y":"oc"}]
    for property in ["phc","phh","ec","caco3","p","n","k"]:
        configs.append({"x":[property], "y":"oc"})
        configs.append({"x":["oc"], "y":property})
        #configs.append({"x":["665", "560", "490"], "y":"oc"})
        configs.append({"x":["665", "560", "490", property], "y":"oc"})
        configs.append({"x":["665", "560", "490"], "y":property})
        configs.append({"x":["665", "560", "490", "oc"], "y":property})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=2,
        prefix="all"
    )
    ev.process()
    print("Done all")
import os
from evaluator import Evaluator


def process():
    configs = []

    for property in ["phc","phh","ec","caco3","p","n","k","stones"]:
        configs.append({"x":[property], "y":"oc"})

    ev = Evaluator(
        cofigs=configs,
        repeat=1,
        folds=10,
        prefix=f"prop_oc"
    )
    ev.process()

    print("Done all")


if __name__ == "__main__":
    os.chdir("../")
    process()
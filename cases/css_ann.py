import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    rgb = ["665", "560", "490"]
    css = ["clay", "sand", "silt"]
    configs = []
    for property in css:
        configs.append({"x":rgb+[property], "y":"oc", "machine":"ann"})
        configs.append({"x":[property], "y":"oc", "machine":"ann"})
    configs.append({"x": rgb+css, "y": "oc", "machine": "ann"})
    configs.append({"x": css, "y": "oc", "machine": "ann"})

    ev = Evaluator(
        cofigs=configs,
        file_name="vis_css",
        repeat=1,
        folds=10,
        prefix="css_ann"
    )
    ev.process()
    print("Done all")
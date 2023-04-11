import numpy as np
import pandas as pd
import ds_manager
import os
from datetime import datetime
import torch
from train import train
from test import test


class Evaluator:
    def __init__(self, cofigs=None, prefix="", verbose=False,
                 repeat=1, folds=10, file_name = None
                 ):
        if file_name is None:
            file_name = "vis"
        self.file_name = file_name
        if cofigs is None:
            cofigs = [{"x":["665", "560", "490"], "y":"oc"}]
        self.configs = cofigs
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"
        self.mean_file = f"results/{prefix}_mean.csv"

        self.summary_index = self.create_summary_index()

        self.details = np.zeros((self.folds*self.repeat, len(self.configs)))
        self.details_index = self.get_details_index()
        self.details_columns = self.get_details_columns()
        self.summary_columns = self.get_summary_columns()

        self.sync_details_file()
        self.create_log_file()

        self.TEST = False
        self.TEST_SCORE = 0

    def get_details_columns(self):
        details_columns = []
        for config in self.configs:
            details_columns.append(f"{self.get_config_name(config)}")
        return details_columns

    def get_summary_columns(self):
        return ["ANN"]

    def get_details_index(self):
        details_index = []
        for i in range(self.repeat):
            for fold in range(self.folds):
                details_index.append(f"I-{i}-{fold}")
        return details_index

    def get_details_row(self, repeat_number, fold_number):
        return self.folds*repeat_number + fold_number

    def set_details(self, index_config, repeat_number, fold_number, score):
        details_row = self.get_details_row(repeat_number, fold_number)
        self.details[details_row,index_config] = score

    def get_details(self, index_config, repeat_number, fold_number):
        details_row = self.get_details_row(repeat_number, fold_number)
        return self.details[details_row,index_config]

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.details = df.to_numpy()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self, summary):
        df = pd.DataFrame(data=summary, columns=self.summary_columns, index=self.summary_index)
        df.to_csv(self.summary_file)

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.details_columns, index=self.details_index)
        df.to_csv(self.details_file)

    def log_scores(self, repeat_number, fold_number, config, score):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{repeat_number} - {fold_number} - {self.get_config_name(config)}\n")
        log_file.write(str(score))
        log_file.write("\n")
        log_file.close()

    @staticmethod
    def get_config_name(config):
        name = "-".join(config["x"])+"_"+config["y"]
        name = name.replace("665-560-490","RGB")
        return name

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

        score_mean = np.mean(self.details, axis=0)
        score_mean = np.round(score_mean, 3)

        self.write_summary(score_mean)

    def process_repeat(self, repeat_number):
        for index_config, config in enumerate(self.configs):
            self.process_config(repeat_number, index_config)

    def process_config(self, repeat_number, index_config):
        config = self.configs[index_config]
        print("Start", f"{repeat_number}:{self.get_config_name(config)}")
        ds = ds_manager.DSManager(folds=self.folds, x=config["x"], y=config["y"], file_name=self.file_name)

        for fold_number, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
            score = self.get_details(index_config, repeat_number, fold_number)
            if score != 0:
                print(f"{repeat_number}-{fold_number} done already")
            else:
                machine = "ann"
                if "machine" in config:
                    machine = config["machine"]
                score = self.calculate_score(train_ds, test_ds, machine)
                self.log_scores(repeat_number, fold_number, config, score)
            if self.verbose:
                print(score)
            self.set_details(index_config, repeat_number, fold_number, score)
            self.write_details()

    def calculate_score(self, train_ds, test_ds, machine):
        if self.TEST:
            self.TEST_SCORE = self.TEST_SCORE + 1
            return self.TEST_SCORE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = train(device, train_ds, machine)
        return test(device, test_ds, model_instance, machine=machine)

    def create_summary_index(self):
        index = []
        for config in self.configs:
            name = self.get_config_name(config)
            index.append(f"{name}")
        return index





if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x":["665", "560", "490"], "y":"oc"},
            # {"x":["665", "560", "490","n"], "y":"oc"},
            # {"x":["665", "560", "490"], "y":"n"},
            # {"x":["665", "560", "490","oc"], "y":"n"},
            # {"x":["n"], "y":"oc"},
            {"x":["oc"], "y":"n"}
        ],
        repeat=3
    )
    ev.process()
    print("Done all")
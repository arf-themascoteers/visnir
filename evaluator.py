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
                 repeat=1, folds=10
                 ):
        if cofigs is None:
            cofigs = [{"x":["665", "560", "490"], "y":"oc"}]
        self.configs = cofigs
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.summary = np.zeros(len(self.configs))
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"
        self.mean_file = f"results/{prefix}_mean.csv"

        self.summary_index = self.create_summary_index()

        self.details = np.zeros((self.folds*self.repeat, len(self.configs)))
        self.details_index = self.get_details_index()
        self.details_columns = self.get_details_columns()
        self.summary_columns = self.get_summary_columns()

        self.sync_summary_file()
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
        self.details[details_row][index_config] = score

    def get_details(self, index_config, repeat_number, fold_number):
        details_row = self.get_details_row(repeat_number, fold_number)
        return self.details[details_row][index_config]

    def sync_summary_file(self):
        if not os.path.exists(self.summary_file):
            self.write_summary()
        df = pd.read_csv(self.summary_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.summary = df.to_numpy()

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

    def write_summary(self):
        df = pd.DataFrame(data=self.summary, columns=self.summary_columns, index=self.summary_index)
        df.to_csv(self.summary_file)

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.details_columns, index=self.details_index)
        df.to_csv(self.details_file)

    def log_scores(self, config, r2s):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{self.get_config_name(config)}\n")
        log_file.write(str(r2s))
        log_file.write("\n")
        log_file.close()

    def set_score(self, index_config, score):
        self.summary[index_config] = score

    def get_score(self, index_config):
        return self.summary[index_config]

    @staticmethod
    def get_config_name(config):
        name = "-".join(config["x"])+"_"+config["y"]
        name = name.replace("665-560-490","RGB")
        return name

    def process(self):
        for index_config, config in enumerate(self.configs):
            self.process_config(index_config)

    def process_config(self, index_config):
        config = self.configs[index_config]

        print("Start", f"{self.get_config_name(config)}")

        if self.get_score(index_config) != 0:
            print(f"{self.get_config_name(config)} Was done already")
        else:
            scores = self.calculate_scores_folds(index_config)
            score_mean = np.round(np.mean(scores), 3)
            scores = np.round(scores, 3)
            self.log_scores(config, scores)
            self.set_score(index_config, score_mean)
            self.write_summary()

    def create_summary_index(self):
        index = []
        for config in self.configs:
            name = self.get_config_name(config)
            index.append(f"{name}")
        return index

    def create_mean_index(self):
        index = []
        for config in self.configs:
            name = self.get_config_name(config)
            index.append(name)
        return index

    def calculate_scores_folds(self, index_config):
        config = self.configs[index_config]
        scores = []
        for repeat_number in range(self.repeat):
            ds = ds_manager.DSManager(folds=self.folds, x=config["x"], y=config["y"])

            for fold_number, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
                score = self.get_details(index_config, repeat_number, fold_number)
                if score != 0:
                    print(f"{repeat_number}-{fold_number} done already")
                else:
                    score = self.calculate_score(train_ds, test_ds)
                if self.verbose:
                    print(score)
                scores.append(score)
                self.set_details(index_config, repeat_number, fold_number, score)
                self.write_details()
        return scores

    def calculate_score(self, train_ds, test_ds):
        if self.TEST:
            self.TEST_SCORE = self.TEST_SCORE + 1
            return self.TEST_SCORE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = train(device, train_ds)
        return test(device, test_ds, model_instance)



if __name__ == "__main__":
    ev = Evaluator(
        cofigs=[
            {"x":["665", "560", "490"], "y":"oc"},
            {"x":["665", "560", "490","n"], "y":"oc"},
            {"x":["665", "560", "490","oc"], "y":"n"},
            {"x":["oc"], "y":"n"},
        ],
        repeat=3
    )
    ev.process()
    print("Done all")
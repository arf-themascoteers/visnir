import numpy as np
import pandas as pd
import ds_manager
import os
from datetime import datetime
import torch
from ann import ANN

class Evaluator:
    def __init__(self, prefix="", verbose=False,
                 repeat=1, folds=5, alpha=0
                 ):
        self.repeat = repeat
        self.alpha = alpha
        self.folds = folds
        self.verbose = verbose

        self.summary_file_n = f"results/{prefix}_summary_n.csv"
        self.details_file_n = f"results/{prefix}_details_n.csv"
        self.summary_file_oc = f"results/{prefix}_summary_oc.csv"
        self.details_file_oc = f"results/{prefix}_details_oc.csv"
        self.log_file = f"results/{prefix}_log.txt"

        self.summary_index = self.create_summary_index()

        self.details_n = np.zeros((self.folds*self.repeat, 1))
        self.details_oc = np.zeros((self.folds*self.repeat, 1))
        self.details_index = self.get_details_index()
        self.details_columns = self.get_details_columns()
        self.summary_columns = self.get_summary_columns()

        self.sync_details_file()
        self.create_log_file()

    def get_details_columns(self):
        details_columns = []
        details_columns.append(f"RUN")
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

    def set_details(self, index_config, repeat_number, fold_number, score_n, score_oc):
        details_row = self.get_details_row(repeat_number, fold_number)
        self.details_n[details_row,index_config] = score_n
        self.details_oc[details_row,index_config] = score_oc

    def get_details(self, index_config, repeat_number, fold_number):
        details_row = self.get_details_row(repeat_number, fold_number)
        return self.details_n[details_row,index_config], self.details_oc[details_row,index_config]

    def sync_details_file(self):
        self.details_n = self.sync_details_file_for_output(self.details_file_n)
        self.details_oc = self.sync_details_file_for_output(self.details_file_oc)

    def sync_details_file_for_output(self, output_file):
        if not os.path.exists(output_file):
            self.write_details()
        df = pd.read_csv(output_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        return df.to_numpy()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self, summary_n, summary_oc):
        self.write_summary_for(summary_n, self.summary_file_n)
        self.write_summary_for(summary_oc, self.summary_file_oc)

    def write_summary_for(self, summary, output):
        df = pd.DataFrame(data=summary, columns=self.summary_columns, index=self.summary_index)
        df.to_csv(output)

    def write_details(self):
        self.write_details_for(self.details_n, self.details_file_n)
        self.write_details_for(self.details_oc, self.details_file_oc)

    def write_details_for(self, output, output_file):
        df = pd.DataFrame(data=output, columns=self.details_columns, index=self.details_index)
        df.to_csv(output_file)

    def log_scores(self, repeat_number, fold_number, config, score_n, score_oc):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{repeat_number} - {fold_number} - RUN\n")
        log_file.write(str(score_n))
        log_file.write(str(score_oc))
        log_file.write("\n")
        log_file.close()

    @staticmethod
    def get_config_name(config):
        name = "-".join(config["x"])
        if "intermediate" in config and config["intermediate"] is not None:
            name = name + "_"+ ("-".join(config["intermediate"]))
        name = name +"_"+config["y"]
        name = name.replace("665-560-490","RGB")
        return name

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

        score_mean_n = np.mean(self.details_n, axis=0)
        score_mean_n = np.round(score_mean_n, 3)
        score_mean_oc = np.mean(self.details_oc, axis=0)
        score_mean_oc = np.round(score_mean_oc, 3)

        self.write_summary(score_mean_n, score_mean_oc)

    def process_repeat(self, repeat_number):
        self.process_config(repeat_number, 1)

    def process_config(self, repeat_number, index_config):
        config = "RUN"
        print("Start", f"{repeat_number}:RUN")
        ds = ds_manager.DSManager(folds=self.folds)

        for fold_number, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
            score_n, score_oc = self.get_details(index_config, repeat_number, fold_number)
            if score_oc != 0:
                print(f"{repeat_number}-{fold_number} done already")
            else:
                score_n, score_oc = self.calculate_score(train_ds, test_ds)
                self.log_scores(repeat_number, fold_number, config, score_n, score_oc)
            if self.verbose:
                print(f"{score_n}--{score_oc}")
            self.set_details(index_config, repeat_number, fold_number, score_n, score_oc)
            self.write_details()

    def calculate_score(self, train_ds, test_ds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ANN(device, train_ds, test_ds, self.alpha)
        model.train_model()
        return model.test()

    def create_summary_index(self):
        index = []
        name = "RUN"
        index.append(f"{name}")
        return index

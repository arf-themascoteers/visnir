import numpy as np
import pandas as pd
import ds_manager
import os
from datetime import datetime
import torch
from train import train
from test import test
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class Evaluator:
    def __init__(self, datasets=None, algorithms=None,
                 colour_space_models=None, prefix="", verbose=False,
                 repeat=1, folds=10, calc_mean = False
                 ):
        self.repeat = repeat
        self.folds = folds
        self.datasets = datasets
        self.calc_mean = calc_mean
        if self.datasets is None:
            self.datasets = ["lucas"]
        self.algorithms = algorithms
        if self.algorithms is None:
            self.algorithms = ["nn"]
        self.colour_spaces = colour_space_models
        if self.colour_spaces is None:
            self.colour_spaces = ["vis"]
        self.colour_space_names = []
        for i in self.colour_spaces:
            if isinstance(i, str):
                self.colour_space_names.append(i)
            elif type(i) is dict:
                self.colour_space_names.append(i["cspace"])
        self.verbose = verbose
        self.summary = np.zeros((len(self.colour_spaces) * len(self.datasets), len(self.algorithms)))
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"
        self.mean_file = f"results/{prefix}_mean.csv"

        self.summary_index = self.create_summary_index()

        self.details = np.zeros((len(self.datasets)*self.folds*self.repeat, len(self.algorithms) * len(self.colour_spaces)))
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
        for algorithm in self.algorithms:
            for colour_space in self.colour_spaces:
                details_columns.append(f"{self.get_alg_name(algorithm)}-{self.get_cspace_name(colour_space)}")
        return details_columns

    def get_summary_columns(self):
        details_columns = []
        for algorithm in self.algorithms:
            details_columns.append(f"{self.get_alg_name(algorithm)}")
        return details_columns


    def get_details_index(self):
        details_index = []
        for index_dataset, dataset in enumerate(self.datasets):
            for i in range(self.repeat):
                for fold in range(self.folds):
                    details_index.append(f"{dataset}-{i}-{fold}")
        return details_index

    def get_details_row(self, index_dataset, itr_no):
        return index_dataset*self.folds*self.repeat + itr_no

    def get_details_column(self, index_algorithm, index_colour_space):
        return len(self.colour_spaces) * index_algorithm + index_colour_space

    def set_details(self, index_algorithm, index_colour_space, index_dataset, it_now, score):
        details_row = self.get_details_row(index_dataset, it_now)
        details_column = self.get_details_column(index_algorithm, index_colour_space)
        self.details[details_row][details_column] = score

    def get_details(self, index_algorithm, index_colour_space, index_dataset, it_now):
        details_row = self.get_details_row(index_dataset, it_now)
        details_column = self.get_details_column(index_algorithm, index_colour_space)
        return self.details[details_row][details_column]

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


    def get_summary_row(self, index_dataset, index_colour_space):
        return (index_dataset*len(self.colour_spaces)) + index_colour_space

    def log_scores(self, dataset, algorithm, colour_space, r2s):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{dataset} - {algorithm} - {colour_space}\n")
        log_file.write(str(r2s))
        log_file.write("\n")
        log_file.close()

    def set_score(self, index_dataset, index_algorithm, index_colour_space, score):
        row = self.get_summary_row(index_dataset, index_colour_space)
        self.summary[row][index_algorithm] = score

    def get_score(self, index_dataset, index_algorithm, index_colour_space):
        row = self.get_summary_row(index_dataset, index_colour_space)
        return self.summary[row][index_algorithm]

    @staticmethod
    def get_cspace_name(colour_space):
        if type(colour_space) is dict:
            if "name" in colour_space:
                if colour_space["name"] is not None:
                    return colour_space["name"]
            else:
                return f"{colour_space['cspace']}"
        if isinstance(colour_space, str):
            return colour_space

    @staticmethod
    def get_alg_type(alg):
        if type(alg) is dict:
            return f"{alg['atype']}"
        if isinstance(alg, str):
            return alg

    @staticmethod
    def get_alg_name(alg):
        if type(alg) is dict:
            if "name" in alg:
                if alg["name"] is not None:
                    return alg["name"]
            else:
                return f"{alg['atype']}"
        if isinstance(alg, str):
            return alg

    @staticmethod
    def get_nn_config(alg):
        if type(alg) is dict:
            return alg
        return {}


    def process_algorithm_colour_space_dataset(self, index_algorithm, index_colour_space, index_dataset):
        algorithm = self.algorithms[index_algorithm]
        colour_space = self.colour_spaces[index_colour_space]
        dataset = self.datasets[index_dataset]

        print("Start", f"{dataset} - {self.get_alg_name(algorithm)} - {self.get_cspace_name(colour_space)}")

        if self.get_score(index_dataset, index_algorithm, index_colour_space) != 0:
            print(f"{dataset} - {algorithm} - {colour_space} Was done already")
        else:
            scores = self.calculate_scores_folds(index_algorithm, index_colour_space, index_dataset)
            score_mean = np.round(np.mean(scores), 3)
            scores = np.round(scores, 3)
            self.log_scores(dataset, algorithm, colour_space, scores)
            self.set_score(index_dataset, index_algorithm, index_colour_space, score_mean)
            self.write_summary()

    def process_algorithm_colour_space(self, index_algorithm, index_colour_space):
        for index_dataset, dataset in enumerate(self.datasets):
            self.process_algorithm_colour_space_dataset(index_algorithm, index_colour_space, index_dataset)

    def process_algorithm(self, index_algorithm):
        for index_colour_space, colour_space in enumerate(self.colour_spaces):
            self.process_algorithm_colour_space(index_algorithm, index_colour_space)

    def process(self):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(index_algorithm)
        self.calculate_mean()

    def create_summary_index(self):
        index = []
        for dataset in self.datasets:
            for colour_space in self.colour_spaces:
                name = self.get_cspace_name(colour_space)

                index.append(f"{dataset} - {name}")
        return index

    def create_mean_index(self):
        index = []
        for colour_space in self.colour_spaces:
            name = self.get_cspace_name(colour_space)
            index.append(name)
        return index

    def calculate_scores_folds(self, index_algorithm, index_colour_space, index_dataset):
        algorithm = self.algorithms[index_algorithm]
        colour_space = self.colour_spaces[index_colour_space]
        dataset = self.datasets[index_dataset]
        scores = []
        for i in range(self.repeat):
            if type(colour_space) is dict:
                ds = ds_manager.DSManager(dataset, **colour_space, random_state=i, folds=self.folds)
            else:
                ds = ds_manager.DSManager(dataset, colour_space, random_state=i, folds=self.folds)

            for itr_no, (train_ds, test_ds) in enumerate(ds.get_k_folds()):
                it_now = i*self.folds + itr_no
                score = self.get_details(index_algorithm, index_colour_space, index_dataset, it_now)
                if score != 0:
                    print(f"{it_now} done already")
                else:
                    score = self.calculate_score(train_ds, test_ds, algorithm)
                if self.verbose:
                    print(score)
                scores.append(score)
                self.set_details(index_algorithm, index_colour_space, index_dataset, it_now, score)
                self.write_details()
        return scores

    def calculate_score(self, train_ds, test_ds, algorithm):
        if self.TEST:
            self.TEST_SCORE = self.TEST_SCORE + 1
            return self.TEST_SCORE

        model_instance = None

        algorithm_type = self.get_alg_type(algorithm)
        nn_config = self.get_nn_config(algorithm)

        if algorithm_type == "nn":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = train(device, train_ds, nn_config = nn_config)
            return test(device, test_ds, model_instance)
        else:
            train_x = train_ds.get_x()
            train_y = train_ds.get_y()
            test_x = test_ds.get_x()
            test_y = test_ds.get_y()

            if algorithm_type == "lr":
                model_instance = LinearRegression()
            elif algorithm_type == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm_type == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm_type == "svr":
                model_instance = SVR()


            model_instance = model_instance.fit(train_x, train_y)
            return model_instance.score(test_x, test_y)

    def calculate_mean(self):
        mean = np.zeros((len(self.colour_spaces), len(self.algorithms)))

        for index_colour_space, colour_space in enumerate(self.colour_spaces):
            for index_algorithm, algorithm in enumerate(self.algorithms):
                mean[index_colour_space, index_algorithm] = \
                    self.calculate_mean_algorithm_colour_space(index_algorithm, index_colour_space)

        df = pd.DataFrame(data=mean, columns=self.summary_columns, index=self.create_mean_index())
        df.to_csv(self.mean_file)

    def calculate_mean_algorithm_colour_space(self, index_algorithm, index_colour_space):
        score = 0
        for index_dataset, dataset in enumerate(self.datasets):
            score = score + self.get_score(index_dataset, index_algorithm, index_colour_space)
        return np.round(score/len(self.datasets),3)


if __name__ == "__main__":
    ev = Evaluator(calc_mean=True)
    ev.process()
    print("Done all")
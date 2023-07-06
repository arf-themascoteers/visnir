from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype


class DSManager:
    def __init__(self, name=None, folds=10, config="rgb"):
        self.file = "data/vis_with_empty.csv"
        train_df, test_df = self.get_random_train_test_df()
        self.x = ["665", "560", "490"]
        self.y = "oc"
        self.name = name
        self.folds = folds
        df = pd.concat([train_df, test_df])

        if config != "rgb":
            self.x = self.x + ["n"]

        columns = self.x + [self.y]
        df = df[columns]
        self.full_data = df.to_numpy()
        self.full_data = self._normalize(self.full_data)
        self.train = self.full_data[0:len(train_df)]
        self.test = self.full_data[len(train_df):]

    def get_random_train_test_df(self):
        df = self.read_from_csv(self.file)
        return model_selection.train_test_split(df, test_size=0.2, random_state=2)

    def read_from_csv(self, file):
        df = pd.read_csv(file)
        return df

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield SpectralDataset(train_data), SpectralDataset(test_data)

    def get_folds(self):
        return self.folds

    def _normalize(self, data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:,i].reshape(-1, 1))
            data[:,i] = np.squeeze(x_scaled)
        return data

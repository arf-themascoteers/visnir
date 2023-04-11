from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype


class DSManager:
    def __init__(self, name=None, folds=10, x=None, y="oc",file_name=None, min_row=0):
        self.min_row = min_row
        if file_name is None:
            file_name = "vis"
        if x is None:
            x = ["665", "560", "490"]
        self.x = x
        self.y = y
        self.name = name
        self.folds = folds

        csv_file_location = f"data/{file_name}.csv"
        df = pd.read_csv(csv_file_location)
        columns = x + [y]
        df = df[columns]
        print(f"Initial rows {len(df)}")
        df = self.process_ohe(df)
        print(f"OHE-Filtered rows {len(df)}")
        npdf = df.to_numpy()
        npdf = self._normalize(npdf)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=2)
        self.full_data = np.concatenate((train, test), axis=0)
        self.full_ds = SpectralDataset(self.full_data)
        self.train_ds = SpectralDataset(train)
        self.test_ds = SpectralDataset(test)

    def process_ohe(self, df):
        newdf = df.copy()

        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                uniques = df[col].unique()
                for a_value in uniques:
                    if len(newdf[newdf[col] == a_value]) < self.min_row:
                        newdf = newdf.drop(newdf[newdf.lc1 == a_value].index)
                y = pd.get_dummies(newdf[col], prefix=col)
                newdf = newdf.drop(col, axis=1)
                newdf = pd.concat([y,newdf], axis=1)

        return newdf

    def get_test_ds(self):
        return self.test_ds

    def get_train_ds(self):
        return self.train_ds

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

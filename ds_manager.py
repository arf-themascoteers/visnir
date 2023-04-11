from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype


class DSManager:
    def __init__(self, name=None, folds=10, x=None, y="oc",file_name=None, min_row=0, intermediate=None):
        if x is None:
            x = ["665", "560", "490"]
        self.x = x
        if intermediate is None:
            intermediate = []
        self.intermediate = intermediate
        self.y = y
        self.name = name

        self.min_row = min_row
        if file_name is None:
            file_name = "vis"

        self.folds = folds

        csv_file_location = f"data/{file_name}.csv"
        df = pd.read_csv(csv_file_location)

        columns = x + intermediate + [y]
        df = df[columns]

        df, ohe_offset = self.process_ohe(df)

        self.x = self.get_updated_columns(df, self.x)
        self.intermediate = self.get_updated_columns(df, self.intermediate)
        npdf = df.to_numpy()
        npdf = self._normalize(npdf, ohe_offset)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=2)
        self.full_data = np.concatenate((train, test), axis=0)
        self.full_ds = SpectralDataset(self.full_data, self.x, self.intermediate)
        self.train_ds = SpectralDataset(train, self.x, self.intermediate)
        self.test_ds = SpectralDataset(test, self.x, self.intermediate)

    def process_ohe(self, df):
        ohe_offset = 0
        newdf = df.copy()
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                uniques = df[col].unique()
                for a_value in uniques:
                    if len(newdf[newdf[col] == a_value]) < self.min_row:
                        newdf = newdf.drop(newdf[newdf[col] == a_value].index)
                y = pd.get_dummies(newdf[col], prefix=col)
                ohe_offset = ohe_offset + len(y.columns)
                newdf = newdf.drop(col, axis=1)
                newdf = pd.concat([y,newdf], axis=1)

        return newdf, ohe_offset

    def get_test_ds(self):
        return self.test_ds

    def get_train_ds(self):
        return self.train_ds

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield SpectralDataset(train_data, self.x, self.intermediate), \
                SpectralDataset(test_data, self.x, self.intermediate)

    def get_folds(self):
        return self.folds

    def _normalize(self, data, offset=0):
        for i in range(offset, data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:,i].reshape(-1, 1))
            data[:,i] = np.squeeze(x_scaled)
        return data

    def get_updated_columns(self, df, columns):
        indices = []
        for index,col in enumerate(df.columns):
            if col in columns:
                indices.append(index)
            else:
                for i in columns:
                    if col.startswith(f"{i}_"):
                        indices.append(index)
        return indices
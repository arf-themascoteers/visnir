from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold


class DSManager:
    def __init__(self, name=None, folds=10, x=None, y="oc"):
        if x is None:
            x = ["665", "560", "490"]
        self.x = x
        self.y = y
        self.name = name
        self.folds = folds

        csv_file_location = f"data/vis.csv"
        df = pd.read_csv(csv_file_location)
        columns = x + [y]
        df = df[columns]
        npdf = df.to_numpy()
        npdf = self._normalize(npdf)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=2)
        self.full_data = np.concatenate((train, test), axis=0)
        self.full_ds = SpectralDataset(self.full_data)
        self.train_ds = SpectralDataset(train)
        self.test_ds = SpectralDataset(test)

    @staticmethod
    def equalize_datasets(ds1: SpectralDataset, ds2: SpectralDataset):
        d1 = ds1.df
        d2 = ds2.df
        size = min(d1.shape[0], d2.shape[0])
        return SpectralDataset(d1[0:size,:]), SpectralDataset(d2[0:size,:])

    @staticmethod
    def minify_datasets(ds: SpectralDataset, factor):
        d = ds.df
        size = d.shape[0]
        size = int(size*factor)
        return SpectralDataset(d[0:size,:])

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
            # if i != data.shape[1]-1:
            #     continue
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:,i].reshape(-1, 1))
            data[:,i] = np.squeeze(x_scaled)
        return data

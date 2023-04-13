from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype


class DSManager:
    def __init__(self, name=None, folds=10, x=None, y="oc",min_row=0, intermediate=None, files=None, ratios=None):
        self.files = files
        if x is None:
            x = ["665", "560", "490"]
        self.x = x
        if ratios is None:
            ratios = (1,1)
        self.ratios = ratios
        if intermediate is None:
            intermediate = []
        self.intermediate = intermediate
        self.y = y
        self.name = name

        self.min_row = min_row

        self.folds = folds

        train_df, test_df = None, None

        if self.files is None:
            train_df, test_df = self.get_random_train_test_df()
        else:
            train_df, test_df = self.get_train_test_df_from_files(self.files[0], self.files[1])

        df = pd.concat([train_df, test_df])
        columns = x + intermediate + [y]
        df = df[columns]
        df, ohe_offset = self.process_ohe(df)
        self.x = self.get_updated_columns(df, self.x)
        self.intermediate = self.get_updated_columns(df, self.intermediate)
        self.full_data = df.to_numpy()
        self.full_data = self._normalize(self.full_data, ohe_offset)
        self.train = self.full_data[0:len(train_df)]
        self.test = self.full_data[len(train_df):]

    def get_random_train_test_df(self):
        csv_file_location = f"data/vis_with_empty.csv"
        df = pd.read_csv(csv_file_location)
        return model_selection.train_test_split(df, test_size=0.2, random_state=2)

    def get_train_test_df_from_files(self, train_file, test_file):
        train_df = pd.read_csv(train_file)
        #train_df = train_df.sample(n=int(len(train_df)*self.ratios[0]))
        test_df = pd.read_csv(test_file)
        #test_df = test_df.sample(n=int(len(test_df) * self.ratios[0]))
        return train_df, test_df


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

    def get_random_by_ratio(self, array:np.ndarray, ratio):
        indices = np.random.choice(array.shape[0], int(array.shape[0]*ratio), replace=False)
        return array[indices]


    def get_k_folds(self):
        if self.files is not None:
            for i in range(self.folds):
                yield SpectralDataset(self.get_random_by_ratio(self.train,self.ratios[0]), self.x, self.intermediate), \
                      SpectralDataset(self.get_random_by_ratio(self.test,self.ratios[1]), self.x, self.intermediate)
        else:
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
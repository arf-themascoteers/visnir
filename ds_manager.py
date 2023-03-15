from spectral_dataset import SpectralDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold


class DSManager:
    rgb_si = ["soci", "ibs", "ikaw", "rgri", "difgb",
                   "difgr", "ColFeatInd", "RI", "CI", "SI",
                   "BI", "HI", "rn", "gn", "bn", "ngrdi", "difNorGB"
                   ]
    hsv_si = ["h", "s", "v"]
    
    def __init__(self, dt, cspace, si=None, si_only=False,
                 normalize = True, random_state=0,
                 name=None, folds=10, clip_soc=0
                 ):
        self.name = name
        self.folds = folds
        self.dt = dt

        if si is None:
            si = []
        csv_file_location = f"data/{dt}/{cspace}.csv"
        df = pd.read_csv(csv_file_location)
        npdf = df.to_numpy()
        npdf = self.process_si(dt, npdf, si, si_only)
        if clip_soc !=0:
            npdf = npdf[npdf[:,-1]<clip_soc]
        if normalize:
            npdf = self._normalize(npdf)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=random_state)
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

    def process_si(self, dt, npdf, si, si_only):
        if len(si) == 0:
            return npdf

        csv_file_location = f"data/{dt}/rgb.csv"
        df = pd.read_csv(csv_file_location)
        rgb = df.to_numpy()[:,0:3]

        si_values = np.zeros((rgb.shape[0], len(si)))

        for index, a_si in enumerate(si):
            a_si_values = self.determine_si(a_si)
            si_values[:, index] = a_si_values

        base_features = npdf[:,0:-1]
        if si_only:
            base_features = si_values
        else:
            base_features = np.concatenate((base_features, si_values), axis=1)
        soc = npdf[:, -1].reshape(-1,1)
        return np.concatenate((base_features, soc), axis=1)

    def determine_si(self, a_si):
        if a_si in DSManager.rgb_si:
            csv_file_location = f"data/{self.dt}/rgb.csv"
            df = pd.read_csv(csv_file_location)
            rgb = df.to_numpy()[:,0:3]
    
            RED = rgb[:,0]
            GREEN =  rgb[:,1]
            BLUE =  rgb[:,2]
    
            if a_si == "soci":
                return (BLUE) / (RED * GREEN)
    
            if a_si == "ibs":
                return 1/(BLUE ** 2)
    
            if a_si == "ikaw":
                return RED - (BLUE) / (RED) + BLUE
    
            if a_si == "rgri":
                return (RED) / (GREEN)
    
            if a_si == "difgb":
                return (GREEN - BLUE)
    
            if a_si == "difgr":
                return (GREEN - RED)



            if a_si == "ColFeatInd":
                return (GREEN / BLUE)
    
            if a_si == "RI":
                return (RED**2) / ((BLUE)*(GREEN)**3)
    
            if a_si == "CI":
                return (RED - GREEN) / (RED + GREEN)
    
            if a_si == "SI":
                return (RED - BLUE) / (RED + BLUE)
    
            if a_si == "BI":
                return np.sqrt( (RED**2) + (GREEN**2))
    
            if a_si == "HI":
                return 2*(RED-GREEN-BLUE) / (GREEN - BLUE)
    
            SUM = RED + BLUE + GREEN
    
            if a_si == "rn":
                return (RED) / (SUM)
    
            if a_si == "gn":
                return (GREEN) / (SUM)
    
            if a_si == "bn":
                return (BLUE) / (SUM)
    
            if a_si == "ngrdi":
                return (GREEN - RED) / (GREEN + RED)

            if a_si == "difNorGB":
                return (GREEN/SUM) - (BLUE/SUM)

        if a_si in DSManager.hsv_si:
            csv_file_location = f"data/{self.dt}/hsv.csv"
            df = pd.read_csv(csv_file_location)
            hsv = df.to_numpy()[:, 0:3]
            
            h = hsv[:,0]
            s = hsv[:,1]
            v = hsv[:,2]

            if a_si == "h":
                return h
            if a_si == "s":
                return s
            if a_si == "v":
                return v

        return None

    @staticmethod
    def si_list():
        return DSManager.rgb_si + DSManager.hsv_si
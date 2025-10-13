import pandas as pd 
import torch
from torch.utils.data import Dataset
import os

class Dataset_image(Dataset):
    def __init__(self, data_path, train=True):
        self.train = train
        self.data_path = data_path
        self.file_path = "D:/data/" + self.data_path + ("_train_grouped.pkl" if self.train else "_test_grouped.pkl")
        if not os.path.exists(self.file_path):
            self._preprocess_data()
        self.df = pd.read_pickle(self.file_path)
        
    def _preprocess_data(self):
        if self.train:
            df = pd.read_pickle("D:/data/" + self.data_path + ".pkl")[:123203]
        else:
            df = pd.read_pickle("D:/data/" + self.data_path + ".pkl")[123203:]
   
        grouped_data = df.groupby('patient').apply(lambda x: (x['label'].tolist(), x['idx'].tolist()))
        pd.to_pickle(grouped_data, self.file_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        patient = self.df.index[idx]
        labels, idxs = self.df[idx]
        imgs = []
        for idx in idxs:
            try:
                imgs.append(pd.read_pickle(f"D:/feature/image_{int(idx)}.pkl"))
            except:
                pass
        imgs = torch.stack(imgs)
        return imgs, labels[0],patient.replace(" ",'_')
    

class Dataset_BreakHis(Dataset):
    def __init__(self, data_path, train=True):
        self.train = train
        self.data_path = data_path
        self.file_path = "D:/data/" + self.data_path + ("_train_grouped.pkl" if self.train else "_test_grouped.pkl")
        if not os.path.exists(self.file_path):
            self._preprocess_data()
        self.df = pd.read_pickle(self.file_path)
        
    def _preprocess_data(self):
        if self.train:
            df = pd.read_pickle("D:/data/" + self.data_path + "_train.pkl")
        else:
            df = pd.read_pickle("D:/data/" + self.data_path + "_test.pkl")
   
        grouped_data = df.groupby('patient').apply(lambda x: (x['label'].tolist(), x['idx'].tolist()))
        pd.to_pickle(grouped_data, self.file_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        labels, idxs = self.df[idx]
        imgs = []

        for idx in idxs:
            try:
                if self.train:
                    imgs.append(pd.read_pickle(f"D:/feature_breakhis_train/image_{int(idx)}.pkl"))
                else:
                    imgs.append(pd.read_pickle(f"D:/feature_breakhis_test/image_{int(idx)}.pkl"))
            except:
                pass
        imgs = torch.stack(imgs)
        return imgs, labels[0]
    




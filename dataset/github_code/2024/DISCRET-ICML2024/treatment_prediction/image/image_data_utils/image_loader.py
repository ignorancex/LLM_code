# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# t1 = "/workspace/data/uganda/Uganda2000_processed/GeoKey1_BAND1.csv"
# t2 = "/workspace/data/uganda/Uganda2000_processed/GeoKey1_BAND2.csv"
# t3 = "/workspace/data/uganda/Uganda2000_processed/GeoKey1_BAND3.csv"
# b1 = np.genfromtxt(t1, delimiter=',')[1:,:].astype(int)
# b2 = np.genfromtxt(t2, delimiter=',')[1:,:].astype(int)
# b3 = np.genfromtxt(t3, delimiter=',')[1:,:].astype(int)
# # img =  np.array([b3,b1,b2])
# # img = img.transpose(1,2,0).astype(int)
# plt.imshow(b1)
# plt.show()
# a=0


import torch
from torch.utils.data import Dataset
import pandas as pd

DEFAULT_DATA_ROOT = "/workspace/data/uganda/"
DEFAULT_IMG_FORMAT = "Uganda2000_processed/GeoKey{}_BAND{}.csv"

# geokey = pd.read_csv(DEFAULT_DATA_ROOT+"UgandaGeoKeyMat.csv")
# keys = geokey["key"].tolist()
# geocodes = pd.read_csv(DEFAULT_DATA_ROOT+"UgandaGeocodes.csv")
# data_processed = pd.read_csv(DEFAULT_DATA_ROOT+"UgandaDataProcessed.csv")

import numpy as np
import pandas as pd
class CausalDataset(Dataset):
    def __init__(self, processed_path = DEFAULT_DATA_ROOT, img_format = DEFAULT_IMG_FORMAT):
        self.processed_path = processed_path
        self.img_format = img_format
        print(self.processed_path+"UgandaGeoKeyMat.csv")
        self.geokey_data = pd.read_csv(self.processed_path+"UgandaGeoKeyMat.csv")
        self.keys = self.geokey_data["key"].tolist()
        self.geocodes = pd.read_csv(self.processed_path+"UgandaGeocodes.csv")
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        bands = {}
        key = self.keys[idx]
        for i in range(1,4):
            path = self.processed_path + self.img_format.format(key,i)
            img = np.genfromtxt(path, delimiter=',')[1:,:].astype(int)
            img_metadata = np.genfromtxt(path, delimiter=',')[0,:].astype(int)
            bands["band{}".format(i)] = img
            bands["img_metadata"] = img_metadata
        geo_long_value, geo_lat_value = self.geokey_data.loc[self.geokey_data['key'] == key, ['geo_long', 'geo_lat']].iloc[0]
        metadata = self.geocodes.loc[(self.geocodes["geo_long"] == geo_long_value) & (self.geocodes["geo_lat"] == geo_lat_value)].iloc[0].to_dict()
        ret = dict(bands, **metadata)
        ret["key"] = key
        return ret

# import os    
# ds = CausalDataset()
# example_image_folder = "/workspace/data/examples/"
# os.makedirs(example_image_folder, exist_ok=True)
# b = ds[0]
# c = 0
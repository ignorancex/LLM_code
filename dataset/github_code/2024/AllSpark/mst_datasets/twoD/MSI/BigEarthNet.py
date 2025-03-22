import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

# S-1 (4dp)
# band:VV` -- mean:-12.6200, std:5.1159
# band:VH -- mean:-19.2904, std:5.4644

# S-2 (4dp)
# band:B01 -- mean:711.44852, std:1564.5292
# band:B02 -- mean:783.1296, std:1562.5066
# band:B03 -- mean:923.6776, std:1436.2452
# band:B04 -- mean:912.0620, std:1527.7863
# band:B05 -- mean:1263.0253, std:1531.7996
# band:B06 -- mean:2044.0737, std:1583.0283
# band:B07 -- mean:2292.6069, std:1630.4586
# band:B08 -- mean:2445.4717, std:1732.2150
# band:B09 -- mean:2445.7380, std:1587.6179
# band:B10 -- mean:1492.5336, std:1080.1777
# band:B11 -- mean:964.3022, std:807.7414
# band:B12 -- mean:2460.2112, std:1644.2445


BigEarthNet_19_labels_mapping = {
    "Urban fabric": 0,
    "Industrial or commercial units": 1,
    "Arable land": 2,
    "Permanent crops": 3,
    "Pastures": 4,
    "Complex cultivation patterns": 5,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
    "Agro-forestry areas": 7,
    "Broad-leaved forest": 8,
    "Coniferous forest": 9,
    "Mixed forest": 10,
    "Natural grassland and sparsely vegetated areas": 11,
    "Moors, heathland and sclerophyllous vegetation": 12,
    "Transitional woodland, shrub": 13,
    "Beaches, dunes, sands": 14,
    "Inland wetlands": 15,
    "Coastal wetlands": 16,
    "Inland waters": 17,
    "Marine waters": 18
}


class BigEarthNetDataset(Dataset):
    def __init__(self, metainfo_file, root_path, is_train=True):
        self.root_path = root_path
        self.data_infos = self.load_annt(metainfo_file)
        self.band_num = 12
        
        # origin size: 140
        # resize: 140 for patch 14 x 14
        self.resize_transform = transforms.Resize(140, interpolation=3)
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[711.44852, 783.1296, 923.6776, 912.0620, 1263.0253, 2044.0737, 
                        2292.6069, 2445.4717, 2445.4717, 2445.7380, 964.3022, 2460.2112], 
                    std=[1564.5292, 1562.5066, 1436.2452, 1527.7863, 1531.7996, 1583.0283,
                        1630.4586, 1732.2150, 1732.2150, 1587.6179, 807.7414, 1644.2445])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize(
                    mean=[711.44852, 783.1296, 923.6776, 912.0620, 1263.0253, 2044.0737, 
                        2292.6069, 2445.4717, 2445.4717, 2445.7380, 964.3022, 2460.2112], 
                    std=[1564.5292, 1562.5066, 1436.2452, 1527.7863, 1531.7996, 1583.0283,
                        1630.4586, 1732.2150, 1732.2150, 1587.6179, 807.7414, 1644.2445])
                ])

        self.prompts = [
            "Identify the land cover types from the 19 BigEarthNet categories present in this Sentinel-2 satellite image",
            "The given Sentinel-2 satellite image data showcases an area. Describe its main land cover characteristics",
            "Given the following Sentinel-2 satellite image features, list the possible land cover types"
        ]
        
        self.results = dict()

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_root = data_info['img_path']
        imgs = os.listdir(img_root)
        imgs.sort()
        
        bands = []
        for img in imgs:
            if not img.endswith(".tif"):
                continue
            temp = Image.open(os.path.join(img_root, img)).convert("F")
            temp = self.resize_transform(temp)
            temp = np.array(temp).astype(np.float32)
            bands.append(temp)
        img_msi = torch.tensor(np.stack(bands, axis=0))
        
        label = data_info['gt_labels']
        
        if self.transform:
            img_msi = self.transform(img_msi)
        return img_msi, label
    
    def eval(self, cls_score, target):
        cls_score = cls_score.detach().cpu()
        if torch.is_tensor(target):
            target = target.detach().cpu()
        
        if 'pred' not in self.results:
            self.results['pred'] = cls_score
        else:
            self.results['pred'] = torch.cat([self.results['pred'], cls_score], dim=0)
            
        if 'target' not in self.results:
            self.results['target'] = target
        else:
            self.results['target'] = torch.cat([self.results['target'], target], dim=0)
            
    
    def get_eval_res(self):
        labels = self.results['target']
        score = self.results['pred']
        
        average_precision = average_precision_score(labels, score, average='micro') * 100.0

        score = (score > 0.5)

        f1 = f1_score(labels, score, average='micro')

        p = precision_score(labels, score, average='micro')
        r = recall_score(labels, score, average='micro')
        f2_score = 5*p*r / (4*p + r)
        
        self.results = dict()

        return {"AP":average_precision, "f1 score":f1, "f2 score":f2_score}
    
    def load_annt(self, metainfo_path):
        '''
        generate data info

        data_infos: [
            {
                img_path: 'sampleN/S2****'
                gt_labels: [0, 1, 0, 0, ...]
            }
            ...
        ]
        '''
        with open(metainfo_path) as f:    
            meta_infos = json.load(f)

        load_path_list = os.listdir(self.root_path)
        data_infos = []
        for sample_name in tqdm(load_path_list):
            info = dict()
            sample_path = os.path.join(self.root_path, sample_name)
            
            if sample_name == "sample59160":
                continue
            
            # only need optical image
            modal = os.listdir(sample_path)
            modal.sort()
            if len(modal) != 2:
                continue
            modal = modal[1]
            assert modal.startswith("S2")

            img_path = os.path.join(sample_path, modal)
            
            file_list = os.listdir(img_path)
            if len(file_list) != 13:
                continue

            meta_info = meta_infos[sample_name]
            labels = np.zeros(19, dtype=np.int8)
            for label in meta_info:
                labels[BigEarthNet_19_labels_mapping[label]] = 1
                
            info['gt_labels'] = labels
            info['img_path'] = img_path
            data_infos.append(info)

        return data_infos

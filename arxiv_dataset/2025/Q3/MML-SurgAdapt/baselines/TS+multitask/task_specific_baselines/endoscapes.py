import os
import pickle
import json
import torch
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random

class Endoscapes(torch.utils.data.Dataset):

    def __init__(self,split,transform=None):

        self.data_path = '/home/swalimbe/datasets'
        self.split = split
        self.transform = transform
        self.data = self.read_data()

    def read_data(self):
        
        data = []

        def read_json(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        
        endo_path = os.path.join(self.data_path,f"endoscapes/{self.split}")
        b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))
        for p in b['images']:
            filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])
            impath = os.path.join(endo_path,filename)
            gt = [round(value) for value in gt]
            video_id = f"endoscapes_{video}"
            data.append([impath,gt,video_id])

        return data

    def __getitem__(self,index):

        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")
        label = torch.Tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label, vid

    def __len__(self):
        return len(self.data)

    def labels(self):
        file = "cholec/endo_labels.txt"
        with open(file, 'r')as f:
            text = f.read()
        return text.split('\n')

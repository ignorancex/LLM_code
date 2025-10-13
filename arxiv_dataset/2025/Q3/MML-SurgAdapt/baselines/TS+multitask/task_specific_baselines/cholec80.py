import os
import pickle
import json
import torch
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random

class Cholec80(torch.utils.data.Dataset):

    def __init__(self,split,transform=None):

        self.data_path = '/home/swalimbe/datasets'
        self.split = split
        self.transform = transform
        self.data = self.read_data()

    def read_data(self):

        data = []

        cholec80_frames = os.path.join(self.data_path,f"cholec80/frames/{self.split}")
        if self.split != 'train':
            cholec80_labels = os.path.join(self.data_path,f"cholec80/labels/{self.split}/1fps.pickle")
        else:
            cholec80_labels = os.path.join(self.data_path,"cholec80/labels/train/1fps_100_0.pickle")
        a = pickle.load(open(cholec80_labels,"rb"))
        if self.split == 'val':
            invalid_80 = [42]
        elif self.split == 'train':
            invalid_80 = [6,10,14,32]
        elif self.split == 'test':
            valid_80 = [51, 53, 54, 55, 58, 59, 61, 63, 64, 69, 73, 74, 76, 77, 80]
        for video in a.keys():
            id = int(video[5:])
            if self.split != 'test':
                if id in invalid_80:
                    continue
            else:
                if id not in valid_80:
                    continue
            video_folder = os.path.join(cholec80_frames,video)
            input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]
            video_id = f"cholec80_{video}"
            for image,gt in input:
                filename = f"{image}.jpg"
                label = torch.zeros(7)
                label[gt] = 1
                impath = os.path.join(video_folder,filename)
                data.append([impath,label,video_id])

        return data

    def __getitem__(self,index):

        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, vid

    def __len__(self):
        return len(self.data)

    def labels(self):
        file = "cholec/phase_labels.txt"
        with open(file, 'r')as f:
            text = f.read()
        return text.split('\n')

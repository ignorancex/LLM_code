import os
import pickle
import json
import torch
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random

class CholecT50(torch.utils.data.Dataset):

    def __init__(self,split,transform=None):

        self.data_path = '/home/swalimbe/datasets'
        self.split = split
        self.transform = transform
        self.data = self.read_data()

    def read_data(self):

        data = []

        if self.split == 'train':
            index_split = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
        elif self.split == 'val':
            index_split = [8,12,29,50,78]
        elif self.split == 'test':
            index_split = [6,10,14,32,42,51,73,74,80,111]

        cholect50_frames = os.path.join(self.data_path,"cholect50/videos")
        cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

        def get_label(labels):

            output = torch.zeros(100)
            for label in labels:
                index = label[0]
                if index == -1:
                    continue
                output[index] = 1
                
            return output

        for i in index_split:
            video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")
            label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")
            video_id = f"cholect50_{i}"

            with open(label_file, 'r') as file:
                a = json.load(file)

            for frame_id,gts in a['annotations'].items():
                filename = f"{int(frame_id):06d}.png"
                impath = os.path.join(video_folder,filename)
                label = get_label(gts)
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
        file = "cholec/triplet_labels.txt"
        with open(file, 'r')as f:
            text = f.read()
        return text.split('\n')
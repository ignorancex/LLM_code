import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from utils import cls_accuracy
from utils.metrics import Avg_values


class NWPURESIS45Dataset(Dataset):
    def __init__(self, metainfo_file, root_path, is_train=True):
        self.metainfo = pd.read_csv(metainfo_file)
        self.root_path = root_path
        self.target_transform = None
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.class_list = [
            'parking_lot', 'mobile_home_park', 'roundabout', 'lake', 'tennis_court', 'forest', 
            'chaparral', 'desert', 'sparse_residential', 'ground_track_field', 'mountain', 
            'ship', 'golf_course', 'industrial_area', 'palace', 'railway_station', 'harbor', 
            'dense_residential', 'beach', 'railway', 'river', 'baseball_diamond', 'stadium', 
            'airport', 'medium_residential', 'basketball_court', 'intersection', 'storage_tank', 
            'overpass', 'wetland', 'island', 'airplane', 'church', 'runway', 'snowberg', 'sea_ice', 
            'cloud', 'terrace', 'freeway', 'thermal_power_station', 'commercial_area', 'bridge', 
            'rectangular_farmland', 'circular_farmland', 'meadow'
        ]

        self.prompts = [
            f"This remote sensing image belongs to which of the following categories: {', '.join(self.class_list)}?",
            "Describe this remote sensing image briefly",
            "Find a word that is most relevant to this remote sensing image",
            "Describe the key elements in this remote sensing image"
        ]
        
        self.results = dict()

    def __len__(self):
        return self.metainfo.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.metainfo.iloc[idx, 1])
        image = Image.open(img_path).convert('RGB')
        label = self.metainfo.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def eval(self, cls_score, target, topk=(1, 5)):
        batch_size = cls_score.size(0)
        res = cls_accuracy(cls_score, target, topk)
        for key in res.keys():
            if key not in self.results.keys():
                self.results[key] = Avg_values()
            self.results[key].update(res[key].item(), batch_size)
    
    def get_eval_res(self):
        res = dict()
        for key in self.results.keys():
            res[key] = self.results[key].avg
        self.results = dict()
        return res

import os
from PIL import Image
import json
from .base_loader import BaseLoader
import csv

class advbench_Subset_Loader(BaseLoader):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']

        image_width = config['image_width']
        image_height = config['image_height']

        if not os.path.isfile(self.dataset_path) :
            raise ValueError(f"dataset_path不存在: {self.dataset_path}")     

        self.data = []
        
        with open(self.dataset_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            next(reader)

            for row in reader:
                _, goal, target, category, Original_index = row
                
                if 'image_path' in config:
                    image = Image.open(config['image_path']).convert('RGB').resize((image_width, image_height))
                
                else:
                    image = None

                data_dict = {
                    'type': category,
                    'target': target,
                    'origin_question': goal,
                    'image': image
                }

                self.data.append(data_dict)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
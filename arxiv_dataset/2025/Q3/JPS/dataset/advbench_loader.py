import os
from PIL import Image
import json
from .base_loader import BaseLoader
import csv

class advbenchLoader(BaseLoader):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']

        if not os.path.isfile(self.dataset_path) :
            raise ValueError(f"dataset_path不存在: {self.dataset_path}")     

        self.data = []
        
        with open(self.dataset_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            next(reader)

            for row in reader:
                image = None

                data_dict = {
                    'type': 'advbench',
                    'target': row[1],
                    'origin_question': row[0],
                    'image': image
                }

                self.data.append(data_dict)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
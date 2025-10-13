import os
from PIL import Image
import json
from .base_loader import BaseLoader
import csv

class harmbenchLoader(BaseLoader):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']

        if not os.path.isfile(self.dataset_path) :
            raise ValueError(f"dataset_path不存在: {self.dataset_path}")     

        self.data = []
        
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                data_dict = {
                    'type': 'harmbench',
                    'target': item['target'],
                    'origin_question': item['query'],
                    'image': None
                }

                self.data.append(data_dict)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
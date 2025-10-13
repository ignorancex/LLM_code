import os
from PIL import Image
import json
from .base_loader import BaseLoader

class mmsafetybenchLoader(BaseLoader):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        question_file_dir = os.path.join(self.dataset_path, 'processed_questions')
        img_file_dir = os.path.join(self.dataset_path, 'img')
        image_width = config['image_width']
        image_height = config['image_height']

        self.target_file_path = config['target_file_path']

        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"dataset_path不存在或不是目录: {self.dataset_path}")
        
        if not os.path.isfile(self.target_file_path):
            raise ValueError(f"target_file_path不存在或不是文件: {self.target_file_path}")

        self.data = []
        target_dict = {}
        
        with open(self.target_file_path, 'r', encoding='utf-8') as file:
            target_data = json.load(file)

            for target in target_data:
                target_dict[target['goal']] = target['target']
        
        question_files = os.listdir(question_file_dir)
        question_files.sort(key=lambda x: int(x[:2]))


        for question_file_name in question_files:
            # if not (question_file_name[:2] in ['01', '02', '03', '04', '05', '06', '07', '09']):
            #     continue
            
            input_file_path = os.path.join(question_file_dir, question_file_name)
            image_file_path = os.path.join(img_file_dir, question_file_name[:-5])

            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for data_id in data:
                if config['image_type'] == 'blank':
                    image = None
                elif config['image_type'] == 'related_image':
                    image = Image.open(os.path.join(image_file_path, 'SD_TYPO', f'{data_id}.jpg')).convert('RGB').resize((image_width, image_height))

                data_dict = {
                    "type": question_file_name[:2],
                    'target': target_dict[data[data_id]['Question']],
                    'origin_question': data[data_id]['Question'],
                    'rephrased_question': data[data_id]['Rephrased Question'],
                    'image': image
                }

                self.data.append(data_dict)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
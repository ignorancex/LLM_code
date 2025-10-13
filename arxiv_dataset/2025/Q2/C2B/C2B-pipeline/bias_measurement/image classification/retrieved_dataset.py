import json
import os

from PIL import Image
from torchvision.datasets import VisionDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'gif']
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])


class ImageNetRetrievedDataset(VisionDataset):
    def __init__(self, dataset_path, transform):
        super().__init__(transform=transform)
        with open('imagenet-simple-labels.json', 'r') as f:
            labels = json.load(f)

        source = dataset_path.split('/')[-1].split('-')[-1]
        if os.path.exists(f'{source}_records.json'):
            with open(f'{source}_records.json', 'r') as f:
                self.records = json.load(f)
        else:
            self.records = []
            for i, target_class in enumerate(labels):
                for bias_attribute in sorted(os.listdir(os.path.join(dataset_path, 'class', target_class))):
                    for bias_class in sorted(os.listdir(os.path.join(dataset_path, 'class', target_class, bias_attribute))):
                        for file in sorted(os.listdir(os.path.join(dataset_path, 'class', target_class, bias_attribute, bias_class))):
                            if file.split('.')[-1] in IMAGE_EXTENSIONS:
                                self.records.append(
                                    {'target': i, 'target class': target_class, 'bias attribute': bias_attribute, 'bias_class': bias_class,
                                     'img_path': os.path.join(dataset_path, 'class', target_class, bias_attribute, bias_class, file)}
                                )
            with open(f'{source}_records.json', 'w') as f:
                json.dump(self.records, f)

    def __getitem__(self, index):
        record = self.records[index]
        img = Image.open(record['img_path']).convert('RGB')
        target = record['target']

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.records)

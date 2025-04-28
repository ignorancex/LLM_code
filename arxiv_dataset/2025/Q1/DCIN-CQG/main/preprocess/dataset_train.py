import json
import os
from PIL import Image, ImageOps
import numpy as np
from pycocotools import mask as coco_mask
import albumentations as A

import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from preprocess.augmentations import basic_transforms, all_augs
from preprocess import constants


class TrainDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        # load the JSON file
        with open(json_path) as f:
            self.data = json.load(f)
        
        # set values
        self.image_dir = image_dir
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.transform = transform
        self.categories = {category['id']: category['name'] for category in self.data['categories']} 
        self.label_map = {cat_id: constants.LABEL_TO_INT[self.categories[cat_id]] for cat_id in self.categories}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load the image
        img_data = self.images[idx]
        img_id = img_data['id']
        img_path = os.path.join(self.image_dir, img_data['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = np.array(image)

        # find annotations for image
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_id]
        
        # create mask
        mask = np.zeros((img_data['height'], img_data['width']), dtype=np.uint8)

        # add each annotation to the mask
        for ann in annotations:
            category_id = ann['category_id']
            segmentation = ann['segmentation']
            
            rles = coco_mask.frPyObjects(segmentation, img_data['height'], img_data['width'])
            rle = coco_mask.merge(rles)
            binary_mask = coco_mask.decode(rle)
            
            mask[binary_mask == 1] = self.label_map[category_id]

        # apply augmentations
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        return image, mask


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, transform_size=(768, 512), augs='Basline'):
        super().__init__()
        self.train_path = constants.TRAIN_PATH
        self.val_path = constants.VAL_PATH
        self.img_dir = constants.IMG_DIR_HQ
        self.batch_size = batch_size
        
        if augs == 'Baseline':
            self.transform = basic_transforms(transform_size[0], transform_size[1])
        elif augs == 'Augs':
            self.transform = all_augs(transform_size[0], transform_size[1])
        self.val_transform = basic_transforms(transform_size[0], transform_size[1])

    def setup(self, stage=None):
        # Initialize train and validation datasets
        self.train_dataset = TrainDataset(self.train_path, self.img_dir, transform=self.transform)
        self.val_dataset = TrainDataset(self.val_path, self.img_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
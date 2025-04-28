import json
import os
from PIL import Image, ImageOps
import numpy as np
from pycocotools import mask as coco_mask

import torch
from torch.utils.data import Dataset

from preprocess.augmentations import basic_transforms
from preprocess.color_transfer import get_sample_references
from preprocess import constants


class ValDataset(Dataset):
    def __init__(self, transform_size=(768, 512), modes=[]):
        # load the JSON file
        json_path = constants.VAL_PATH
        with open(json_path) as f:
            self.data = json.load(f)
        
        # set values
        self.image_dir = constants.IMG_DIR_HQ
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = {category['id']: category['name'] for category in self.data['categories']} 
        self.label_map = {cat_id: constants.LABEL_TO_INT[self.categories[cat_id]] for cat_id in self.categories}

        self.ref_imgs = get_sample_references()
        self.transform = basic_transforms(transform_size[0], transform_size[1])

        self.modes = modes

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

        images = [image]

        # apply augmentations
        if self.transform:
            transformed_images = []
            transformed_mask = mask
            
            for img in images:
                aug = self.transform(image=img, mask=mask)
                transformed_images.append(aug['image'])
                transformed_mask = aug['mask']

            images = transformed_images
            mask = transformed_mask
        
        return torch.stack(images), mask
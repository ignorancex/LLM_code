import json
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

from preprocess.augmentations import basic_transforms, basic_degrade
from preprocess.color_transfer import open_img, get_sample_references, color_transfer_lab, color_transfer_MKL, color_transfer_sot
from preprocess.lris_utils import load_model_and_db, search_ref_image
from preprocess import constants


class TestDataset(Dataset):
    def __init__(self, dataset='lq', transform_size=(768, 512), modes=[]):
        # load the JSON file
        if dataset == 'lq':
            json_path = constants.LQ_PATH
        else:
            json_path = constants.SP_PATH
        with open(json_path) as f:
            self.data = json.load(f)
        
        # set values
        if dataset == 'lq':
            self.image_dir = constants.IMG_DIR_LQ
        else:
            self.image_dir = constants.IMG_DIR_SP
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = {category['id']: category['name'] for category in self.data['categories']} 
        self.label_map = {cat_id: constants.LABEL_TO_INT[self.categories[cat_id]] for cat_id in self.categories}

        self.ref_imgs = get_sample_references()
        self.transform = basic_transforms(transform_size[0], transform_size[1])
        if 'degrade' in modes:
            self.transform = basic_degrade(transform_size[0], transform_size[1])

        self.modes = modes

        self.ref_histograms = []
        for ref_img in self.ref_imgs:
            ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2Lab)
            ref_hist = cv2.calcHist([ref_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
            self.ref_histograms.append(ref_hist)

        self.feat_encoder, self.feat_vectors, self.feat_ref_imgs = load_model_and_db(model_name='swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
                                                                                     db_root='/feature_vector_DB/extracted_features')

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

        # color transfer
        images = []
        if "dcin_gris" in self.modes:
            images = [color_transfer_lab(image, ref_img) for ref_img in self.ref_imgs]
        elif "dcin_lris" in self.modes:
            best_path = search_ref_image(img_path, self.feat_encoder, self.feat_vectors, self.feat_ref_imgs, 384)
            best_ref = open_img(best_path)
            images = [color_transfer_lab(image, best_ref)]
        elif "dcin_full" in self.modes:
            images = [color_transfer_lab(image, ref_img) for ref_img in self.ref_imgs]
            best_path = search_ref_image(img_path, self.feat_encoder, self.feat_vectors, self.feat_ref_imgs, 384)
            best_ref = open_img(best_path)
            images += [color_transfer_lab(image, best_ref)]
        else:
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

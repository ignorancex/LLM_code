import numpy as np
import random
import torch
import copy
import time

from PIL import Image
from torchvision.transforms import Resize, Normalize
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter

from cxrvlms.modeling.constants import *

BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'

# Augmentations for pretraining
augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))


# Function lo load an image from data dict, and scale to target size
def load_image(image_path, size, canvas):

    # Read image
    if "dcm" in image_path or "dicom" in image_path:
        import pydicom
        dicom = pydicom.read_file(image_path)
        img = np.array(dicom.pixel_array, dtype=float)
        # Intensity normalization
        if np.max(img) > 255 or np.min(img) < 0:
            img = (img - np.min(img)) / (np.max(img-np.min(img)) / 255.0)
        # Negative required for some datasets
        if "VinDr" in image_path:
            img = -img + 255
    else:
        img = Image.open(image_path)
        max_size = max(img.size)
        scale = max_size / size[0]
        img.draft('L', (img.size[0] / scale, img.size[1] // scale))
        img = np.asarray(img, dtype=float)

    if (len(img.shape) > 2) and (img.shape[-1] < 5):
        img = img[:, :, 0]

    # Scale intensity
    img /= 255.

    # Add channel
    img = np.expand_dims(img, 0)

    # Resize image
    img = torch.tensor(img)
    if not canvas or (img.shape[-1] == img.shape[-2]):
        img = Resize(size)(img)
    else:
        sizes = img.shape[-2:]
        max_size = max(sizes)
        scale = max_size / size[0]
        img = Resize((int(img.shape[-2] / scale), int((img.shape[-1] / scale)))).cuda()(img.cuda())
        img = torch.nn.functional.pad(img,
                                      (0, size[0] - img.shape[-1], 0, size[1] - img.shape[-2], 0, 0))
    img = img.cpu().numpy()
    return img


def norm_image(img, norm):
    # Add channels to grayscale image
    if img.shape[0] == 1:
        img = np.repeat(img, 3, 0)

    if norm:
        img = torch.tensor(img)
        img = Normalize(mean=[0.5] * 3, std=[0.5] * 3)(img)
        img = img.numpy()

    return img


class LoadImage():

    def __init__(self, size=(224, 224), canvas=True, norm=False):
        self.size = size
        self.canvas = canvas
        self.norm = norm

    def __call__(self, data, cache=False):
        d = copy.deepcopy(data)

        if cache:  # If cache allowed, load image and return, if required.
            if "cache" in d.keys():
                # Retrieve image from cache
                img = np.float32(d["image"]) / 255.

                # Norm Image
                d["image"] = norm_image(img, self.norm)

                # Return updated dict
                return d
            else:
                # Load image
                img = load_image(data['image_path'], self.size, self.canvas)

                # Prepare storing image
                img_storing = np.uint8((img * 255))

                # Norm Image
                d["image"] = norm_image(img, self.norm)

                # Return updated dict
                return d, img_storing

        else:  # Online data loading
            # Load image
            img = load_image(data['image_path'], self.size, self.canvas)
            # Norm Image
            d["image"] = norm_image(img, self.norm)
            return d


class ProduceDescription():
    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):
        if "MIMIC" in data["image_path"]:

            # Sample one of the available text prompts
            idx = random.sample(list(np.arange(0, len(data["prompts"]))), 1)[0]

            # Select text prompt
            data["prompt_selected"] = [data["prompts"][idx]]

            # Select categories for the sentence report from CheXpert-labeler
            category_prompt_selected = np.array(
                [iCategory in data['prompts_categories'][idx] for iCategory in CATEGORIES], dtype=int)
            data["category_prompt_selected"] = category_prompt_selected

            # Create labels mask
            mask_categories = np.array(
                [iCategory in CATEGORIES_DATASETS["MIMIC"] for iCategory in CATEGORIES], dtype=int)
            data["mask_categories"] = mask_categories

        elif "PadChest" in data["image_path"]:

            # If there is no labeled category, create generic text prompt - but this not count for label alignment!
            if len(data["study_categories"]) == 0:
                data["study_categories"] = ["Other finding"]

            # Sample one category for the text prompt
            idx = random.sample(list(np.arange(0, len(data["study_categories"]))), 1)[0]

            # Apply prompt template to category name
            data["prompt_selected"] = [self.caption.replace("[CLS]",  data["study_categories"][idx])]

            # Select category for text (only one for CheXpert)
            category_prompt_selected = np.array(
                [iCategory == data["study_categories"][idx] for iCategory in CATEGORIES], dtype=int)
            data["category_prompt_selected"] = category_prompt_selected

            # Create labels mask
            mask_categories = np.array(
                [iCategory in CATEGORIES_DATASETS["PADCHEST"] for iCategory in CATEGORIES], dtype=int)
            data["mask_categories"] = mask_categories

        elif "CheXpert" in data["image_path"]:

            # If there is no labeled category, create generic text prompt - but this not count for label alignment!
            if len(data["study_categories"]) == 0:
                data["study_categories"] = ["Other finding"]

            # Sample one category for the text prompt
            idx = random.sample(list(np.arange(0, len(data["study_categories"]))), 1)[0]

            # Apply prompt template to category name
            data["prompt_selected"] = [self.caption.replace("[CLS]",  data["study_categories"][idx])]

            # Select category for text (only one for CheXpert)
            category_prompt_selected = np.array(
                [iCategory == data["study_categories"][idx] for iCategory in CATEGORIES], dtype=int)
            data["category_prompt_selected"] = category_prompt_selected

            # Create labels mask
            mask_categories = np.array(
                [iCategory in CATEGORIES_DATASETS["CHEXPERT"] for iCategory in CATEGORIES], dtype=int)
            data["mask_categories"] = mask_categories

        # Get study categories: all categories labelled from any text report
        data["study_categories"] = np.array([iCategory in data["study_categories"] for iCategory in CATEGORIES], dtype=int)

        return data


class SelectRelevantKeys():
    def __call__(self, data):
        d = {key: data[key] for key in ['image', 'study_categories', 'prompt_selected', 'category_prompt_selected',
                                        'mask_categories']}
        return d
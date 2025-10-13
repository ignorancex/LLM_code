import json

import PIL
import numpy as np
import os

from data.utils import _get_img_from_path
from data.abc import AbstractBaseDataset

_DEFAULT_CIRR_DATASET_ROOT = 'data'

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')


class AbstractBaseCIRRDataset(AbstractBaseDataset):
    """
    following the previous standard method
    """
    @classmethod
    def code(cls):
        return 'cirr'

    @classmethod
    def all_codes(cls):
        return ['cirr']


class CIRRDataset(AbstractBaseCIRRDataset):
    """
       CIRR dataset class which manage CIRR data
    """
    def __init__(self, root_path=_DEFAULT_CIRR_DATASET_ROOT, clothing_type='dress', split='train',
                 img_transform=None, text_transform=None, id_transform=None, test_split="samples", mode="classic"):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param test_split: test split, should be in ['samples', 'query']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        """
        self.root_path = root_path
        # self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.id_transform = id_transform
        self.test_split = test_split
        self.mode = mode

        if split not in ['test1', 'train', 'val'] or mode not in ['relative', 'classic']:
            raise ValueError("split should be in ['test1', 'train', 'val'] and mode should be in ['relative', 'classic']")
        # get triplets made by (reference_image, target_image, relative caption)
        with open(self.root_path + "/cirr_dataset/cirr/captions/cap.rc2.{}.json".format(split)) as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(self.root_path + "/cirr_dataset/cirr/image_splits/split.rc2.{}.json".format(split)) as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} initialized")

    def __getitem__(self, idx):
        reference_name = self.triplets[idx]['reference']
        reference_image_path = os.path.join(self.root_path, 'cirr_dataset', self.name_to_relpath[reference_name])
        reference_img = _get_img_from_path(reference_image_path, self.img_transform)

        rel_caption = self.triplets[idx]['caption']
        modifier = caption_post_process(rel_caption)
        modifier = self.text_transform(modifier) if self.text_transform is not None else modifier
        # test1 split no target
        if self.split == "test1":
            if self.mode == "relative":
                group_members = self.triplets[idx]['img_set']['members']
                pair_id = self.triplets[idx]['pairid']
                return pair_id, reference_name, modifier, group_members
            if self.mode == "classic":
                image_name = list(self.name_to_relpath.keys())[idx]
                image_path = os.path.join(self.root_path, 'cirr_dataset', self.name_to_relpath[image_name])
                image = _get_img_from_path(image_path, self.img_transform)
                return image_name, image
        target_hard_name = self.triplets[idx]['target_hard']
        target_image_path = os.path.join(self.root_path, 'cirr_dataset', self.name_to_relpath[target_hard_name])
        target_img = _get_img_from_path(target_image_path, self.img_transform)

        ref_id = reference_name
        targ_id = target_hard_name
        # following the previous data setting, return different things.
        if self.split == "train":
            return reference_img, target_img, modifier, len(modifier), ref_id, targ_id
        if self.split == "val":
            if self.test_split == "samples":
                if self.mode == "classic":
                    image_name = list(self.name_to_relpath.keys())[idx]
                    image_path = os.path.join(self.root_path, 'cirr_dataset', self.name_to_relpath[image_name])
                    image = _get_img_from_path(image_path, self.img_transform)
                    return image, image_name
                return target_img, targ_id
            if self.test_split == "query":
                pair_id = self.triplets[idx]['pairid']
                return reference_img, ref_id, modifier, targ_id, pair_id


    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
if __name__ == '__main__':
    cirr = CIRRDataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data", split="val",
                         mode="classic")
    print(len(cirr), cirr[0])
    cirr_val_relative = CIRRDataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data", split="val",
                            mode="relative")
    print(len(cirr_val_relative), cirr_val_relative[0])

    cirr_test_classic = CIRRDataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data", split="test1",
                            mode="classic")
    print(len(cirr_test_classic), cirr_test_classic[0])
    cirr_test_relative = CIRRDataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data", split="test1",
                            mode="relative")
    print(len(cirr_test_relative), cirr_test_relative[0])

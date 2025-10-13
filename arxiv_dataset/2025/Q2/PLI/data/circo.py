import json
from pathlib import Path
import PIL
import PIL.Image
import numpy as np
import os
from typing import List, Optional, Union, Dict, Literal
from torch.utils.data import Dataset

from data.utils import _get_img_from_path

_DEFAULT_CIRCO_DATASET_ROOT = 'data/CIRCO'
class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, root_path: Union[str, Path]=_DEFAULT_CIRCO_DATASET_ROOT, split: Literal['val', 'test']= 'val',
                 mode: Literal['relative', 'classic']='relative', img_transform=None, text_transform=None, test_split="samples"):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            test_split (str): test split, should be in ['samples', 'query']
                similar function as mode, keep for compatibility
        """
        super().__init__()
        # Set dataset paths and configurations
        dataset_path = Path(root_path)
        self.mode = mode
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.data_path = dataset_path
        self.test_split = test_split

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")
        # align to the CIRCO dataset
        # self.prompt = "a photo of "

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index):
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - ['reference_img', 'reference_img_id', 'target_img', 'modifier', 'shared_concept', 'gt_img_ids', 'query_id']
        """
        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            modifier = self.text_transform(relative_caption) if self.text_transform is not None else relative_caption
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = _get_img_from_path(reference_img_path, self.img_transform)

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = _get_img_from_path(target_img_path, self.img_transform)

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))
                if self.test_split == "samples":
                    return target_img, target_img_id
                if self.test_split == "query":
                    return reference_img, reference_img_id, modifier, target_img_id, query_id

            elif self.split == 'test':
                return reference_img, reference_img_id, modifier, shared_concept, query_id
        if self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = _get_img_from_path(img_path, self.img_transform)
            return img, img_id
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    @classmethod
    def code(cls):
        return 'circo'

# test
if __name__ == '__main__':
    circo = CIRCODataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data/CIRCO", split="val", mode="classic")
    print(len(circo), circo[0])
    circo_2 = CIRCODataset(root_path="/data/chenjy/code/TGIR/Diffusion4TGIR/data/CIRCO", split="val", mode="relative")
    print(len(circo_2), circo_2[0])
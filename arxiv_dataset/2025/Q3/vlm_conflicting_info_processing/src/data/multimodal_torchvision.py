import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import os
import glob
from PIL import Image
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


from utils.prompt_evaluation_utils import (
    get_caption_label,
    get_candidate_labels,
    get_format_labels,
    get_prompt,
)

CUB_DIR = ""
IMAGENET100_DIR = ""


def cub_load_data(data_root, split):
    # Load image data
    images = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "images.txt"),
        sep=" ", names=["image_id", "filepath"],
    )
    image_class_labels = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "image_class_labels.txt"),
        sep=" ", names=["image_id", "class_id"],
    )
    train_test_split = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "train_test_split.txt"),
        sep=" ", names=["image_id", "is_training_image"],
    )
    classes = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "classes.txt"),
        sep=" ", names=["class_id", "class_name"],
    )

    data = images.merge(image_class_labels, on="image_id")
    data = data.merge(train_test_split, on="image_id")
    data = data.merge(classes, on="class_id")
    
    # Get data split
    if split=="train":
        data = data[data.is_training_image==1]
    elif split=="valid":
        data = data[data.is_training_image==0]
    elif split=="all":
        data = data

    data["class_name"] = [class_name.split(".")[1].lower().replace("_", " ") for class_name in data.class_name]

    # Load attribute data
    image_attribute_labels = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"),
        sep=" ", names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
    )
    attributes = pd.read_csv(
        os.path.join(data_root, "CUB_200_2011", "attributes", "attributes.txt"),
        sep=" ", names=["attribute_id", "attribute_name"]
    )
    attributes_info = [attr.split("::") for attr in attributes.attribute_name]
    attributes_info = np.array([[attr.replace("_", " "), label.replace("_", " ")] for attr, label in attributes_info])
    attributes["attribute_template"] = attributes_info[:, 0]
    attributes["attribute_label"] = attributes_info[:, 1]
    attributes = image_attribute_labels.merge(attributes, on="attribute_id")
    unique_attributes = attributes.attribute_template.unique()
    return data, attributes, unique_attributes


class CUB_PrimaryColor(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: Root path,
        split: "train" or "val"
        """
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"

        self.root_dir = root_dir

        data, attributes, _ = cub_load_data(root_dir, split=split)
        all_data = pd.merge(data, attributes, how="inner", on="image_id")

        # Filter only relevant rows once
        filtered = all_data[
            (all_data['attribute_template'] == "has primary color") &
            (all_data['is_present'] == 1)
        ]

        # Group by image_id and count unique attribute_labels
        unique_counts = filtered.groupby('image_id')['attribute_label'].nunique()

        # Keep only image_ids where there's exactly one unique label
        keep_ids = unique_counts[unique_counts == 1].index.tolist()

        filtered_all_data = all_data[all_data['image_id'].isin(keep_ids)]
        self.data_df = filtered_all_data[(filtered_all_data['is_present'] == 1) & (filtered_all_data['attribute_template'] == "has primary color")]

        self.classes = ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey',\
                  'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white',\
                  'red', 'buff']
        self.class_to_idx = {class_name: index for index,class_name in enumerate(self.classes)}

        print(f"CUB Primary Color [{split.upper()}] Loaded {len(self.data_df)} images from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.data_df)
    
    def _get_img(self, img_path):
        return Image.open(f"{self.root_dir}/CUB_200_2011/images/" + img_path)

    def __getitem__(self, idx):
        data_slice = self.data_df.iloc[idx]

        image = self._get_img(data_slice['filepath'])
        label = data_slice['attribute_label']

        return image, self.classes.index(label)

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: Root path
        split: "train" or "val"
        """
        assert split in ["train", "val"], "split must be 'train' or 'val'"

        # load labels
        labels_path = os.path.join(root_dir, "Labels.json")
        with open(labels_path, "r") as f:
            self.labels_dict = json.load(f)
        
        for k in self.labels_dict:
            self.labels_dict[k] = self.labels_dict[k].split(",")[0]
        
        self.classes = list(self.labels_dict.values())
        self.class_to_idx = {class_name: index for index,class_name in enumerate(self.classes)}

        # Match train.X*, val.X*, etc.
        folder_pattern = os.path.join(root_dir, f"{split}.X*")
        split_dirs = sorted(glob.glob(folder_pattern))

        self.image_paths = []
        self.class_names = []

        for split_dir in split_dirs:
            for class_name in sorted(os.listdir(split_dir)):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.class_names.append(class_name)

        print(f"ImageNet100 [{split.upper()}] Loaded {len(self.image_paths)} images from {len(set(self.class_names))} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_idx = self.class_to_idx[self.labels_dict[self.class_names[idx]]]
        image = Image.open(image_path).convert("RGB")
        return image, label_idx


class SingleObjectVOC(Dataset):
    def __init__(self, root, split='train'):
        # Load original VOCDetection dataset
        self.voc = torchvision.datasets.VOCDetection(root=root, image_set=split)  # WARNING: set `download=True` for the first time. This is a torchvision bug.

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
        self.class_to_idx = {class_name: index for index,class_name in enumerate(self.classes)}
        self._raw_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Filter for samples with exactly one object
        self.filtered = []
        for img, target in tqdm(self.voc, total=len(self.voc), desc="Filtering PASCAL"):
            objects = target['annotation'].get('object', [])
            
            if isinstance(objects, list) and len(objects) == 1:
                self.filtered.append((img, objects[0]['name']))

        print(f"Loaded {len(self.filtered)} single-object images from VOC 2012 ({split})")

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, idx):

        image, raw_label = self.filtered[idx]

        return image, self._raw_classes.index(raw_label)


class MultimodalDataset(Dataset):
    def __init__(self, args, split, prompt_generator, simple_caption=False):
        self.args = args
        self.split = split
        self.prompt_generator = prompt_generator

        self.simple_caption = simple_caption
        
        self.data = self._get_data()
        self.classes = self._get_classes()
        self.idx_to_class = self._get_idx_to_class()

        self.all_sources, self.all_choices, self.all_image_labels, self.all_caption_labels, self.all_relative_image_pos, self.all_relative_caption_pos = self._process_data()

    def __len__(self):

        return len(self.data)

    def _get_classes(self):

        if self.args.dataset in ["cifar10", "imagenet100", "Pascal", "CUB_color"]:
            return self.data.classes
        elif self.args.dataset in ["cifar100"]:
            return [c.replace("_", " ") for c in self.data.classes]
        else:
            raise NotImplementedError(f"{self.args.dataset} dataset not supported yet")

    def _get_idx_to_class(self):
        
        if self.args.dataset in ["cifar10", "imagenet100", "Pascal", "CUB_color"]:
            return {v:k for k, v in self.data.class_to_idx.items()}
        elif self.args.dataset in ["cifar100"]:
            return {v:k.replace("_", " ") for k, v in self.data.class_to_idx.items()}
        else:
            raise NotImplementedError(f"{self.args.dataset} dataset not supported yet")


    def _get_data(self):

        if self.args.dataset in ["cifar10"]:
            return torchvision.datasets.CIFAR10(
                root=f"{self.args.work_dir}/datasets",
                train=True if self.split == "train" else False,
                download=True,
            )
        elif self.args.dataset in ["cifar100"]:
            return torchvision.datasets.CIFAR100(
                root=f"{self.args.work_dir}/datasets",
                train=True if self.split == "train" else False,
                download=True,
            )
        elif self.args.dataset in ["imagenet100"]:
            return ImageNet100Dataset(
                root_dir=IMAGENET100_DIR,
                split= "train" if self.split == "train" else "val"
            )
        elif self.args.dataset in ["Pascal"]:
            return SingleObjectVOC(f"{self.args.work_dir}/.cache", 
                split= "train" if self.split == "train" else "val"
            )
        elif self.args.dataset in ["CUB_color"]:
            return CUB_PrimaryColor(
                root_dir=CUB_DIR,
                split= "train" if self.split == "train" else "valid"
            )
        else:
            raise NotImplementedError(f"{self.args.dataset} dataset not supported yet")

    def convert_id_to_label(self, label_id):
        return self.idx_to_class[label_id]

    def _process_data(self):
        all_sources, all_choices, all_image_labels, all_caption_labels = [],[],[],[]
        all_relative_image_pos, all_relative_caption_pos = [], []
        for ex_id, example in tqdm(enumerate(self.data), total=len(self), desc=f"{self.args.dataset} {self.split}: "):

            image, image_label = example 
            caption_label = get_caption_label(self.args, self.classes, image_label)

            if self.simple_caption:
                candidate_labels, im_rel_pos, cap_rel_pos = [], -1, -1
                format_labels = None
                prompt = get_prompt(self.args, None, format_labels, self.classes, caption_label, simple_caption=self.simple_caption)
            else:
                candidate_labels, im_rel_pos, cap_rel_pos = get_candidate_labels(self.args, self.classes, image_label, caption_label)
                format_labels = get_format_labels(self.args, candidate_labels, self.classes, caption_label)

                prompt = get_prompt(self.args, self.prompt_generator, format_labels, self.classes, caption_label, simple_caption=self.simple_caption)

            all_sources.append(prompt)
            all_choices.append(candidate_labels)
            all_image_labels.append(image_label)
            all_caption_labels.append(caption_label)
            all_relative_image_pos.append(im_rel_pos)
            all_relative_caption_pos.append(cap_rel_pos)

        return all_sources, all_choices, all_image_labels, all_caption_labels, all_relative_image_pos, all_relative_caption_pos
 
    def get_image(self, idx):
        return self.data[idx][0]

    def get_choice_ids(self, choices):
        if self.args.dataset in ["cifar10", "imagenet100", "Pascal", "CUB_color"]:
            return [
                self.data.class_to_idx[c]
                for c in choices
            ]
        elif self.args.dataset in ["cifar100"]:
            return [
                self.data.class_to_idx[c.replace(" ", "_")]
                for c in choices
            ]
        else:
            raise NotImplementedError(f"{self.args.dataset} dataset not supported yet")

    def __getitem__(self, idx):
        choices = self.all_choices[idx]
        choice_ids = self.get_choice_ids(choices)
        return {
            "source": self.all_sources[idx],
            "choices": choices,
            "choice_ids": choice_ids,
            "image_label": self.all_image_labels[idx],
            "caption_label": self.all_caption_labels[idx],
            "relative_image_label": self.all_relative_image_pos[idx], 
            "relative_caption_label": self.all_relative_caption_pos[idx],
            "image": self.get_image(idx),
        }


class LmEvaluationDataCollator:
    def __init__(self, processor, is_multiple_choice, text_only=False):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.is_text_only = text_only
    

    def __call__(self, features, return_tensors="pt"):
        raw_texts = [feature["source"] for feature in features]
        images    = [feature["image"] for feature in features]

        if self.is_text_only:

            inputs = self.processor.tokenizer(
                raw_texts, 
                padding=True,
                return_tensors="pt"
            )

        else:
            inputs = self.processor(
                text=raw_texts, 
                images=images, 
                padding=True,
                return_tensors="pt"
            )


        image_labels = torch.tensor([feature["image_label"] for feature in features])
        caption_labels = torch.tensor([feature["caption_label"] for feature in features])
        choice_ids = torch.tensor([feature["choice_ids"] for feature in features])

        return {
            "inputs": {**inputs},
            "image_labels": image_labels,  # all classes, 0-9
            "caption_labels": caption_labels,  # all classes, 0-9
            "choice_ids": choice_ids,  # all classes, 0-9
        }

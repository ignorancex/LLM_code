import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from q2l_labeller.data.coco_dataset import CoCoDataset
from q2l_labeller.data.cutmix import CutMixCollator
from q2l_labeller.data.dataset import SeaThruAugmentation
import numpy as np
import random
from torch.utils.data import WeightedRandomSampler
import cv2
from PIL import Image

# 1. Define a custom CLAHE transform:
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of the grid for histogram equalization.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply CLAHE on.
        Returns:
            PIL Image after CLAHE.
        """
        # Convert from PIL Image to numpy array
        img_np = np.array(img)

        # If it's a single-channel (grayscale) image:
        if len(img_np.shape) == 2:
            img_np = self.clahe.apply(img_np)
        else:
            # For color images, convert to LAB and apply CLAHE on the L channel only
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert back to PIL Image
        return Image.fromarray(img_np)

def compute_sample_weights(labels):
    """
    Compute sample weights for a multi-label dataset.

    Args:
        labels (numpy.ndarray): Multi-hot encoded labels for the dataset.

    Returns:
        list: List of weights for each sample.
    """
    class_counts = np.sum(labels, axis=0)  # Sum across samples for each class
    class_weights = 1.0 / np.clip(class_counts, 1, None)  # Avoid division by zero
    sample_weights = np.dot(labels, class_weights)  # Compute weights
    return sample_weights.tolist()

class COCODataModule(pl.LightningDataModule):
    """Datamodule for Lightning Trainer"""

    def __init__(
        self,
        data_dir,
        img_size,
        num_classes,  # ✅ Pass dynamic number of classes
        batch_size=128,
        num_workers=0,
        use_cutmix=False,
        cutmix_alpha=1.0,
        train_classes=None,
        augmentation_strategy="baseline",
        seathru_transform=None,
        sampling_strategy="oversample",
        combine_prob=0.5
    ) -> None:
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate
        self.train_classes = train_classes
        self.augmentation_strategy = augmentation_strategy
        self.seathru_transform = seathru_transform
        self.combine_prob = combine_prob
        self.num_classes = num_classes  # ✅ Store dynamic class number

    def setup(self, stage=None) -> None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.augmentation_strategy == "seathru" and self.seathru_transform:
            print("🚀 Depth-Jitter Augmentation Initialized")
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop(self.img_size, padding=8),
                transforms.RandomErasing(p=0.2),
                RandAugment(),
                self.seathru_transform,
                transforms.ToTensor(),
                normalize,
            ])
        elif self.augmentation_strategy == "combined" and self.seathru_transform:
            print("🚀 Combined Augmentation Initialized")
            def combined_transform(image_name, image):
                if torch.rand(1).item() <= self.combine_prob:
                    image = self.seathru_transform(image_name, image)
                return transforms.Compose([
                    RandAugment(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])(image)

            train_transforms = combined_transform
        elif self.augmentation_strategy == "colorjitter":
            print("🚀 ColorJitter Augmentation Initialized")
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.augmentation_strategy == "clahe":
            print("🚀 CLAHE Augmentation Initialized")
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),  #CLAHE transform
                transforms.ToTensor(),
                normalize,
            ])
        else:
            print("🚀 Baseline Augmentation Initialized")
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ])

        test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        # ✅ Pass num_classes when initializing datasets
        if stage == 'fit' or stage is None:
            self.train_set = CoCoDataset(
                image_dir=os.path.join(self.data_dir, "train"),
                anno_path=os.path.join(self.data_dir, "train.json"),
                num_classes=self.num_classes,  # ✅ Dynamically adjust class numbers
                input_transform=train_transforms,
                labels_path=os.path.join(self.data_dir, "annotations/train.npy"),
                train_classes=self.train_classes,
                seathru_transform=self.seathru_transform
            )
            self.sample_weights = compute_sample_weights(np.array(self.train_set.labels))

        self.val_set = CoCoDataset(
            image_dir=os.path.join(self.data_dir, "val"),
            anno_path=os.path.join(self.data_dir, "val.json"),
            num_classes=self.num_classes,  # ✅ Dynamically adjust class numbers
            input_transform=test_transforms,
            labels_path=os.path.join(self.data_dir, "annotations/val.npy"),
            train_classes=self.train_classes,
            seathru_transform=self.seathru_transform
        )

        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

        if self.train_classes is not None:
            self.train_set.filter_samples(self.train_classes)

    def get_num_classes(self):
        """✅ Returns number of classes dynamically"""
        return self.num_classes  # Fix incorrect `self.classes`

    def train_dataloader(self) -> DataLoader:
        if self.sampling_strategy == "oversample":
            sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )

import os
from glob import glob

import numpy as np
from numpy import ndarray

from utils.augmentations import combine_aug, binary_combine_aug
from utils.preprocessing import global_extraction


class MRIDataset:
    non_cropped_dataset_path: str | None
    binary_dataset_path: str | None
    cropped_dataset_path: str | None

    def __init__(self, non_cropped_dataset_path: str = None, binary_dataset_path: str = None,
                 cropped_dataset_path: str = None):
        if non_cropped_dataset_path is not None:
            self.non_cropped_dataset_path = os.path.realpath(non_cropped_dataset_path)
            if not os.path.isdir(self.non_cropped_dataset_path):
                raise NotADirectoryError("The provided path for the non cropped dataset is not a valid directory")

            for subdirectory in ["train/images", "train/masks", "val/images", "val/masks"]:
                subdirectory_path = os.path.join(self.non_cropped_dataset_path, subdirectory)
                if not os.path.isdir(subdirectory_path):
                    raise FileNotFoundError(f"Unable to find the subdirectory {subdirectory}")
                elif len(glob(os.path.join(subdirectory_path, "*.npy"))) == 0:
                    raise FileNotFoundError(f"The {subdirectory} subdirectory is data (when looking for .npy files)")
        else:
            self.non_cropped_dataset_path = None

        if binary_dataset_path is not None:
            self.binary_dataset_path = os.path.realpath(binary_dataset_path)
            if not os.path.isdir(self.binary_dataset_path):
                raise NotADirectoryError("The provided path for the binary dataset is not a valid directory")

            for subdirectory in ["train/images", "train/masks", "val/images", "val/masks"]:
                subdirectory_path = os.path.join(self.binary_dataset_path, subdirectory)
                if not os.path.isdir(subdirectory_path):
                    raise FileNotFoundError(f"Unable to find the subdirectory {subdirectory}")
                elif len(glob(os.path.join(subdirectory_path, "*.npy"))) == 0:
                    raise FileNotFoundError(f"The {subdirectory} subdirectory is data (when looking for .npy files)")
        else:
            self.binary_dataset_path = None

        if cropped_dataset_path is not None:
            self.cropped_dataset_path = os.path.realpath(cropped_dataset_path)
            if not os.path.isdir(self.cropped_dataset_path):
                raise NotADirectoryError("The provided path for the cropped dataset is not a valid directory")

            for subdirectory in ["train/images", "train/masks", "val/images", "val/masks"]:
                subdirectory_path = os.path.join(self.cropped_dataset_path, subdirectory)
                if not os.path.isdir(subdirectory_path):
                    raise FileNotFoundError(f"Unable to find the subdirectory {subdirectory}")
                elif len(glob(os.path.join(subdirectory_path, "*.npy"))) == 0:
                    raise FileNotFoundError(f"The {subdirectory} subdirectory is data (when looking for .npy files)")
        else:
            self.cropped_dataset_path = None

    def binary_train_datagen(self, batch_size: int) -> tuple[ndarray, ndarray]:
        if self.binary_dataset_path is None:
            raise ValueError("The binary dataset isn't defined")

        all_image_paths = glob(os.path.join(self.binary_dataset_path, "train", "images", "*.npy"))
        all_mask_paths = glob(os.path.join(self.binary_dataset_path, "train", "masks", "*.npy"))

        image_mask_path_pairs = list(zip(all_image_paths, all_mask_paths))

        while True:
            batch_start = 0
            batch_end = batch_size

            np.random.shuffle(image_mask_path_pairs)

            all_image_paths, all_mask_paths = map(list, zip(*image_mask_path_pairs))

            while batch_start < len(image_mask_path_pairs):
                upper_bound = min(batch_end, len(image_mask_path_pairs))

                image_batch = np.array([np.load(path) for path in all_image_paths[batch_start:upper_bound]])
                mask_batch = np.array([np.load(path) for path in all_mask_paths[batch_start:upper_bound]])

                image_batch, mask_batch = binary_combine_aug(image_batch, mask_batch)

                batch_start += batch_size
                batch_end += batch_size

                yield image_batch, mask_batch

    def binary_val_datagen(self, batch_size: int) -> tuple[ndarray, ndarray]:
        if self.binary_dataset_path is None:
            raise ValueError("The binary dataset isn't defined")

        all_image_paths = glob(os.path.join(self.binary_dataset_path, "val", "images", "*.npy"))
        all_mask_paths = glob(os.path.join(self.binary_dataset_path, "val", "masks", "*.npy"))

        image_mask_path_pairs = list(zip(all_image_paths, all_mask_paths))

        while True:
            batch_start = 0
            batch_end = batch_size

            all_image_paths, all_mask_paths = map(list, zip(*image_mask_path_pairs))

            while batch_start < len(image_mask_path_pairs):
                upper_bound = min(batch_end, len(image_mask_path_pairs))

                image_batch = np.array([np.load(path) for path in all_image_paths[batch_start:upper_bound]])
                mask_batch = np.array([np.load(path) for path in all_mask_paths[batch_start:upper_bound]])

                batch_start += batch_size
                batch_end += batch_size

                yield image_batch, mask_batch

    def binary_steps_per_epoch(self, batch_size: int) -> tuple[int, int]:
        if self.binary_dataset_path is None:
            raise ValueError("The binary dataset isn't defined")

        all_train_images = glob(os.path.join(self.binary_dataset_path, "train", "images", "*.npy"))
        all_val_images = glob(os.path.join(self.binary_dataset_path, "val", "images", "*.npy"))

        return len(all_train_images) // batch_size, len(all_val_images) // batch_size

    def cropped_train_datagen(self, batch_size: int) -> tuple[ndarray, ndarray]:
        if self.cropped_dataset_path is None:
            raise ValueError("The cropped dataset isn't defined")

        all_image_paths = glob(os.path.join(self.cropped_dataset_path, "train", "images", "*.npy"))
        all_mask_paths = glob(os.path.join(self.cropped_dataset_path, "train", "masks", "*.npy"))

        image_mask_path_pairs = list(zip(all_image_paths, all_mask_paths))

        while True:
            batch_start = 0
            batch_end = batch_size

            np.random.shuffle(image_mask_path_pairs)

            all_image_paths, all_mask_paths = map(list, zip(*image_mask_path_pairs))

            while batch_start < len(image_mask_path_pairs):
                upper_bound = min(batch_end, len(image_mask_path_pairs))

                # Add remove the mask from the image
                image_batch = [np.load(path)[..., :-1] for path in all_image_paths[batch_start:upper_bound]]
                mask_batch = [np.load(path) for path in all_mask_paths[batch_start:upper_bound]]

                image_batch, mask_batch = global_extraction(image_batch, mask_batch)
                image_batch, mask_batch = combine_aug(image_batch, mask_batch)

                batch_start += batch_size
                batch_end += batch_size

                yield image_batch, mask_batch

    def cropped_val_datagen(self, batch_size: int) -> tuple[ndarray, ndarray]:
        if self.cropped_dataset_path is None:
            raise ValueError("The cropped dataset isn't defined")

        all_image_paths = glob(os.path.join(self.cropped_dataset_path, "val", "images", "*.npy"))
        all_mask_paths = glob(os.path.join(self.cropped_dataset_path, "val", "masks", "*.npy"))

        while True:
            batch_start = 0
            batch_end = batch_size

            while batch_start < len(all_image_paths):
                upper_bound = min(batch_end, len(all_image_paths))

                # Add remove the mask from the image
                image_batch = [np.load(path)[..., :-1] for path in all_image_paths[batch_start:upper_bound]]
                mask_batch = [np.load(path) for path in all_mask_paths[batch_start:upper_bound]]

                image_batch, mask_batch = global_extraction(image_batch, mask_batch)

                batch_start += batch_size
                batch_end += batch_size

                yield image_batch, mask_batch

    def cropped_steps_per_epoch(self, batch_size: int) -> tuple[int, int]:
        if self.cropped_dataset_path is None:
            raise ValueError("The cropped dataset isn't defined")

        all_train_images = glob(os.path.join(self.cropped_dataset_path, "train", "images", "*.npy"))
        all_val_images = glob(os.path.join(self.cropped_dataset_path, "val", "images", "*.npy"))

        return len(all_train_images) // batch_size, len(all_val_images) // batch_size

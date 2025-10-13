import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset

from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Optional, Any

from src.data.datamodule import DataModule
from src.utils import PatchExtractionFailedException
from src.utils.utils import call_background_classifier


class BaseBirdsDataModule(DataModule):
    def __init__(self, data_dir, mask_dir, im_size, batch_size, background_classifier: str,
                 order_background_labels: bool = False,
                 background_images_dir: str = None,
                 patch_bg_size=.25,
                 train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__(background_classifier, order_background_labels)
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.im_size = im_size
        self.batch_size_per_device = batch_size
        self.background_images_dir = background_images_dir
        self.save_path = './data/databirds_cache' + ('_background' if background_images_dir else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.im_size),
            transforms.ToTensor()
        ])
        self.patch_bg_size = patch_bg_size

    def prepare_data(self) -> None:
        super().prepare_data()
        if os.path.exists(self.save_path):
            self.load_data()
        else:
            self.fg_images, self.bg_images, self.masks, self.fg_labels, self.train_test_split = self.get_waterbirds_data()
            self.save_data()
        self.fg_images = self.fg_images * 255
        self.bg_images = self.bg_images * 255
        self.masks = self.masks.int()

    def save_data(self):
        data = {
            'fg_images': self.fg_images,
            'bg_images': self.bg_images,
            'masks': self.masks,
            'fg_labels': self.fg_labels,
            'train_test_split': self.train_test_split
        }
        torch.save(data, self.save_path)

    def load_data(self):
        data = torch.load(self.save_path)
        self.fg_images = data['fg_images']
        self.bg_images = data['bg_images']
        self.masks = data['masks']
        self.fg_labels = data['fg_labels']
        self.train_test_split = data['train_test_split']

    def teardown(self, stage: str) -> None:
        super().teardown(stage)

    def assign_bg_labels(self):
        if self.background_classifier:
            resizer = transforms.Resize(self.im_size)
            self.bg_fg_labels = call_background_classifier(self.background_classifier, resizer(self.fg_images),
                                                           self.patch_bg_size, True)
            self.bg_bg_labels = call_background_classifier(self.background_classifier, resizer(self.bg_images),
                                                           self.patch_bg_size, False)
        else:
            self.bg_fg_labels = torch.tensor([0] * len(self.fg_images))
            self.bg_bg_labels = torch.tensor([0] * len(self.bg_images))

    def read_images_in_subfolders(self, root_folder: str, mask_folder: str, target_places=None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_list, mask_list, labels, train_test_split = [], [], [], []
        label_counter = 0

        split_file = os.path.join(os.path.dirname(root_folder), 'train_val_test_split.txt')
        images_file = os.path.join(os.path.dirname(root_folder), 'images.txt')
        split_info = {}
        with open(split_file, 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split()
                split_info[int(image_id)] = int(is_train)

        image_paths = {}
        with open(images_file, 'r') as f:
            for line in f:
                image_id, image_path = line.strip().split()
                image_paths[image_path] = int(image_id)

        for subdir, _, files in sorted(os.walk(root_folder)):
            if subdir == root_folder:
                continue
            if not target_places or any(place.lower() in subdir.split(os.sep)[-1].lower() for place in target_places):
                for file in sorted(files):
                    file_path = os.path.join(subdir, file)
                    mask_path = os.path.join(mask_folder, subdir.split('/')[-1], file.split('.')[0] + '.png')
                    try:
                        with Image.open(file_path) as img:
                            tensor_img = self.transform(img)
                            if tensor_img.shape[0] != 3:
                                continue

                        with Image.open(mask_path) as mask:
                            tensor_mask = self.transform(mask)
                            if tensor_mask.shape[0] != 1:
                                continue

                        image_id = file_path.split('/')[-2:][0] + '/' + file_path.split('/')[-2:][1]

                        mask_list.append(tensor_mask)
                        image_list.append(tensor_img)
                        labels.append(label_counter)
                        train_test_split.append(split_info[image_paths[image_id]])  # Default to test if ID not found
                    except Exception as e:
                        print(f"Error processing {file_path} or {mask_path}: {e}")
                label_counter += 1

        tensor_images = torch.stack(image_list)
        tensor_masks = torch.stack(mask_list)
        labels = torch.tensor(labels)
        train_test_split = torch.tensor(train_test_split)

        return tensor_images, tensor_masks, labels, train_test_split

    def read_background_images(self, background_images_path):
        target_places = ['ocean', 'lake_natural']
        bg_images = []
        for subdir, _, files in sorted(os.walk(background_images_path)):
            if any(place in subdir.split(os.sep)[-1] for place in target_places):
                if subdir == background_images_path:
                    continue
                for file in sorted(files):
                    file_path = os.path.join(subdir, file)
                    with Image.open(file_path) as img:
                        tensor_img = self.transform(img)
                        if tensor_img.shape[0] != 3:
                            continue
                    bg_images.append(tensor_img)
        return torch.stack(bg_images)

    def get_non_intersecting_patch(self, image: torch.Tensor, mask: torch.Tensor):
        _, width, height = image.shape

        patch_width = int(width * self.patch_bg_size)
        patch_height = int(height * self.patch_bg_size)

        left, top = self.get_patch_outside_mask(mask, patch_width, patch_height)
        patch = F.crop(image, top, left, patch_height, patch_width)
        patch = F.resize(patch, [height, width])

        fg = patch * (1 - mask) + (mask) * image

        left_other, top_other = self.get_patch_outside_mask(mask, patch_width, patch_height)
        patch_other = F.crop(image, top_other, left_other, patch_height, patch_width)
        bg = F.resize(patch_other, [height, width])

        return fg, bg

    def get_patch_outside_mask(self, mask, patch_width, patch_height):
        counter = 0
        _, width, height = mask.shape
        left = top = 0
        threshold = 200
        while counter < threshold:
            left = torch.randint(0, width - patch_width, (1,)).item()
            top = torch.randint(0, height - patch_height, (1,)).item()

            patch_box = [top, left, top + patch_height, left + patch_width]
            intersecting = self.check_intersection(mask, patch_box)

            if not intersecting:
                break
            counter += 1
        if left == 0 and top == 0 or counter == threshold:
            raise PatchExtractionFailedException("")
        return left, top

    def check_intersection(self, mask, patch_box):
        mask_patch = F.crop(mask, patch_box[0], patch_box[1], patch_box[2] - patch_box[0], patch_box[3] - patch_box[1])
        return torch.any(mask_patch > 0)

    def get_waterbirds_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images, masks, fg_labels, train_test_split = self.read_images_in_subfolders(self.data_dir, self.mask_dir,
                                                                                    None)
        masks[masks < .1] = 0
        masks[masks >= .1] = 1
        if not self.background_images_dir:
            bg_patches = []
            for i in range(len(images)):
                try:
                    _, bg = self.get_non_intersecting_patch(images[i], masks[i])
                except PatchExtractionFailedException as e:
                    continue
                bg_patches.append(bg)
            while len(bg_patches) < len(images):
                bg_patches.append(bg_patches[torch.randint(0, len(bg_patches), (1,))[0].item()])
            bg_images = torch.stack(bg_patches)
        else:
            bg_images = self.read_background_images(self.background_images_dir)
            bg_images = bg_images[torch.randint(0, len(bg_images), (len(images),))]
        return images, bg_images, masks, fg_labels, train_test_split


class BirdsDataModule(BaseBirdsDataModule):
    def __init__(self, data_dir, mask_dir, im_size, batch_size, background_classifier: str,
                 order_background_labels: bool = False,
                 train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__(data_dir, mask_dir, im_size, batch_size, background_classifier,
                         order_background_labels, train_val_test_split, num_workers, pin_memory)


class PatchBirdsDataModule(BaseBirdsDataModule):
    def __init__(self, data_dir, mask_dir, im_size, batch_size, background_classifier: str, patch_bg_size:float,
                 order_background_labels: bool = False,
                 train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
                 num_workers: int = 0,
                 pin_memory: bool = False, n_patches: int = 5):

        super().__init__(data_dir, mask_dir, im_size, batch_size, background_classifier,
                         order_background_labels, None,patch_bg_size, train_val_test_split, num_workers, pin_memory)
        self.n_patches = n_patches
        self.already_done = False

    def prepare_data(self) -> None:
        super().prepare_data()
        # Additional preparation specific to PatchBirdsDataModule if needed
        pass

    def setup(self, stage=None):
        super().setup(stage)
        if not self.already_done:
            self.data_train = self.create_patches_dataset(torch.stack([batch[0] for batch in self.data_train]),
                                                          torch.stack([batch[2] for batch in self.data_train]))
            self.data_val = self.create_patches_dataset(torch.stack([batch[0] for batch in self.data_val]),
                                                        torch.stack([batch[2] for batch in self.data_val]))
            self.data_test = self.create_patches_dataset(torch.stack([batch[0] for batch in self.data_test]),
                                                         torch.stack([batch[2] for batch in self.data_test]))
            self.already_done = True

    def create_patches_dataset(self, images: torch.Tensor, masks: torch.Tensor):
        all_patches = []
        for i in range(len(images)):
            patches = []
            for j in range(self.n_patches):
                try:
                    _, patch = self.get_non_intersecting_patch(images[i], masks[i])
                    patches.append(patch)
                except PatchExtractionFailedException:
                    break
            if len(patches) == self.n_patches:
                all_patches.append(torch.stack(patches))
        all_patches = torch.stack(all_patches)  # Shape: (num_images, n_patches, channels, height, width)
        return TensorDataset(all_patches)


class CompositeBirdsDataModule(LightningDataModule):
    def __init__(self, data_dir, im_size, batch_size,
                 train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.im_size = im_size
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.data_train = self.data_val = self.data_test = None

    def prepare_data(self) -> None:
        super().prepare_data()
        pass

    def setup(self, stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if not self.data_train and not self.data_val and not self.data_test:
            image_list, labels, train_test_split = [], [], []
            label_counter = 0
            root_folder = self.data_dir
            split_file = os.path.join(os.path.dirname(root_folder), 'train_val_test_split.txt')
            images_file = os.path.join(os.path.dirname(root_folder), 'images.txt')

            split_info = {}
            with open(split_file, 'r') as f:
                for line in f:
                    image_id, is_train = line.strip().split()
                    split_info[int(image_id)] = int(is_train)

            image_paths = {}
            with open(images_file, 'r') as f:
                for line in f:
                    image_id, image_path = line.strip().split()
                    image_paths[image_path] = int(image_id)

            for subdir, _, files in sorted(os.walk(root_folder)):
                if subdir == root_folder:
                    continue
                for file in sorted(files):
                    file_path = os.path.join(subdir, file)
                    try:
                        transform = transforms.Compose([
                            transforms.Resize(self.im_size),
                            transforms.ToTensor()
                        ])

                        with Image.open(file_path) as img:
                            tensor_img = transform(img)
                            if tensor_img.shape[0] != 3:
                                continue

                        image_id = file_path.split('/')[-2:][0] + '/' + file_path.split('/')[-2:][1]

                        image_list.append(tensor_img)
                        labels.append(label_counter)
                        train_test_split.append(split_info[image_paths[image_id]])  # Default to test if ID not found
                    except Exception as e:
                        print(f"Error processing {file_path} : {e}")
                label_counter += 1

            tensor_images = torch.stack(image_list)
            labels = torch.tensor(labels)
            train_test_split = torch.tensor(train_test_split)

            dataset = TensorDataset(tensor_images, labels)
            train_indices = [i for i, is_train in enumerate(train_test_split) if is_train == 1]
            test_indices = [i for i, is_train in enumerate(train_test_split) if is_train == 0]

            train_dataset = Subset(dataset, train_indices)
            self.data_test = Subset(dataset, test_indices)
            train_percentage, val_percentage, test_percentage = self.hparams.train_val_test_split
            self.data_train, self.data_val = random_split(train_dataset,
                                                          [train_percentage, val_percentage + test_percentage],
                                                          torch.Generator().manual_seed(42))

    def set_logger(self, logger):
        self.logger = logger

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

import os
from typing import Optional

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import random
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter

from src.data.datamodule import DataModule
from src.utils import otsu_threshold
from src.utils.utils import normalize_images, call_background_classifier


class SSSDataModule(DataModule):
    def __init__(self, data_dir, background_classifier, order_background_labels, train_val_test_split, num_workers,
                 pin_memory, crop_size, im_size, batch_size=16, zeta=100):
        super().__init__(background_classifier, order_background_labels)
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.zeta = zeta
        self.im_size = im_size
        self.crop_size = crop_size
        self.bg_images_for_fg_images = None

    def assign_bg_labels(self, images):
        if self.background_classifier:
            images = normalize_images(images)
            return call_background_classifier(self.background_classifier, images,
                                              None, False, data_type=np.float32)
        else:
            return torch.ones(len(images))

    def setup(self, stage=None):

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_dir = os.path.join(self.data_dir, "train")
            test_dir = os.path.join(self.data_dir, "test")

            train_images = sorted(
                [os.path.join(train_dir, "images", f) for f in os.listdir(os.path.join(train_dir, "images"))])
            train_labels = sorted(
                [os.path.join(train_dir, "labels", f) for f in os.listdir(os.path.join(train_dir, "labels"))])

            test_images = sorted(
                [os.path.join(test_dir, "images", f) for f in os.listdir(os.path.join(test_dir, "images"))])
            test_labels = sorted(
                [os.path.join(test_dir, "labels", f) for f in os.listdir(os.path.join(test_dir, "labels"))])

            self.fg_images, self.bg_images, self.bg_for_fg_images, self.masks = self._process_patches_for_training(train_images,
                                                                                                      train_labels)

            self.bg_images = self.bg_images[torch.randint(0, len(self.bg_images), (len(self.fg_images),))]

            train_percentage, val_percentage, test_percentage = self.hparams.train_val_test_split
            self.bg_fg_labels = self.assign_bg_labels(self.bg_for_fg_images)
            self.bg_bg_labels = self.assign_bg_labels(self.bg_images)
            if self.fg_labels is None:
                self.fg_labels = torch.tensor([0] * len(self.fg_images))
            dataset = TensorDataset(self.fg_images, self.bg_images, self.masks,
                                    self.bg_bg_labels, self.bg_fg_labels,
                                    self.fg_labels)
            self.data_train, self.data_val = random_split(dataset,
                                                          [train_percentage, val_percentage + test_percentage],
                                                          torch.Generator().manual_seed(42))

            if self.order_background_labels:
                self.order()

            images, masks, source_image_labels, patches_sizes, patches_indices, band_info, filenames = self._process_patches_for_testing(test_images, test_labels)
            self.data_test = GroupedImageDataset(images, masks, source_image_labels, patches_sizes, patches_indices, band_info, filenames)

            self.log_histograms()
            self.plot_images_per_background_label(colormap='BuPu_r')


    def extract_patches_with_stride(self, image, label, patch_size, stride, zero_threshold=None):
        """
        Extracts overlapping patches from an image and label using a given stride.

        Args:
            image (Tensor): Input image tensor of shape (C, H, W).
            label (Tensor): Corresponding label tensor of shape (1, H, W).
            patch_size (tuple): (height, width) of the patches.
            stride (tuple): (vertical stride, horizontal stride).
            zero_threshold (float, optional): Threshold for rejecting patches with too many zeros.

        Returns:
            Tensor: Stacked image patches (N, C, patch_size, patch_size).
            Tensor: Stacked label patches (N, 1, patch_size, patch_size).
            Tensor: Starting indices for each patch in the image (N, 2).
            Tensor: Size of each patch (N, 2).
        """
        _, h, w = image.shape  # Get image dimensions
        patch_h, patch_w = patch_size
        stride_h, stride_w = stride
        
        # Compute patch starting positions
        row_starts = list(range(0, h - patch_h + 1, stride_h))
        col_starts = list(range(0, w - patch_w + 1, stride_w))
        
        # Ensure last patch reaches the boundary
        if row_starts[-1] + patch_h < h:
            row_starts.append(h - patch_h)
        if col_starts[-1] + patch_w < w:
            col_starts.append(w - patch_w)
        
        image_patches = []
        label_patches = []
        patch_indices = []
        
        for row in row_starts:
            for col in col_starts:
                image_patches.append(image[:, row:row + patch_h, col:col + patch_w].unsqueeze(0))
                label_patches.append(label[:, row:row + patch_h, col:col + patch_w].unsqueeze(0))
                patch_indices.append([row, col])
        
        image_patches = torch.cat(image_patches, dim=0)
        label_patches = torch.cat(label_patches, dim=0)
        patch_indices = torch.tensor(patch_indices)
        
        # Tensor to represent the size of each patch
        patch_sizes = torch.full((patch_indices.shape[0], 2), fill_value=0, dtype=torch.long)
        patch_sizes[:, 0] = patch_h  # Height
        patch_sizes[:, 1] = patch_w  # Width
        
        # Optionally reject patches with too many zeros
        if zero_threshold is not None:
            non_zero_mask = (image_patches == 0).float().mean(dim=(1, 2, 3)) < zero_threshold
            image_patches = image_patches[non_zero_mask]
            label_patches = label_patches[non_zero_mask]
            patch_indices = patch_indices[non_zero_mask]
            patch_sizes = patch_sizes[non_zero_mask]
        
        image_patches = self.clip(image_patches)
        image_patches = TF.resize(image_patches, self.im_size, antialias=True)
        label_patches = TF.resize(label_patches, self.im_size, antialias=True, interpolation=InterpolationMode.NEAREST)
        
        return image_patches, label_patches, patch_indices, patch_sizes




    def find_nearest_background(self, fg_indices, bg_indices, bg_patches):
        """
        Finds the nearest background patch for each foreground patch based on patch indices.

        Args:
            fg_indices (Tensor): (N_fg, 2) Foreground patch indices (row, col).
            bg_indices (Tensor): (N_bg, 2) Background patch indices (row, col).
            bg_patches (Tensor): (N_bg, C, patch_size, patch_size) Background patches.

        Returns:
            Tensor: (N_fg, C, patch_size, patch_size) Matched background patches.
        """
        # Compute squared Euclidean distances between foreground and background patches
        dists = torch.cdist(fg_indices.float(), bg_indices.float(), p=2)  # (N_fg, N_bg)

        # Find the index of the nearest background patch for each foreground patch
        nearest_bg_idx = torch.argmin(dists, dim=1)  # (N_fg,)

        return bg_patches[nearest_bg_idx]


    def _process_patches_for_training(self, image_paths, label_paths):


        fg_images, bg_images, bg_for_fg_images, masks, source_image_labels = [], [], [], [], []
        source_counter = 0

        for img_path, lbl_path in zip(image_paths, label_paths):
            image = Image.open(img_path).convert("RGB")
            label = Image.open(lbl_path).convert("L")

            image = (TF.to_tensor(image)[:1, ...]* 255).to(torch.uint8)

            label = (TF.to_tensor(label)[:1, ...] * 255).to(torch.uint8)

            # Identify the blind band using the first channel of the image.
            blind_band_start, blind_band_end = self.identify_blind_band(image[0])
            for image,label in [(image[...,0:blind_band_start], label[...,0:blind_band_start]), 
            (torch.flip(image[...,blind_band_end:],dims=[-1]),torch.flip(label[...,blind_band_end:],dims=[-1]))]:
                if image.float().mean() == 0:
                    # some images are read all zeros although they are not. Not able to solve this problem so I'm just skipping for now
                    continue

                fg_images_per_image, bg_images_per_image, bg_for_fg_patches_per_image, masks_per_image = self.extract_patches(image,label,self.crop_size,500)
                if len(fg_images_per_image) > 0 and len(bg_for_fg_patches_per_image) > 0:
                    fg_images.append(fg_images_per_image)
                    bg_for_fg_images.append(bg_for_fg_patches_per_image)
                    masks.append(masks_per_image)
                if len(bg_images_per_image) > 0:
                    bg_images.append(bg_images_per_image)
        fg_images = torch.cat(fg_images)
        bg_images = torch.cat(bg_images)
        bg_for_fg_images = torch.cat(bg_for_fg_images)
        masks = torch.cat(masks)
        return fg_images, bg_images , bg_for_fg_images, masks 


    def identify_blind_band(self, image: torch.Tensor):
        """
        Identify the blind band (no patching/cropping zone) in a sonar image using PyTorch.
        
        The blind band is defined as the contiguous range of columns for which
        the sum over rows is zero.
        
        Args:
            image (torch.Tensor): A 2D tensor representing the sonar image (H x W).
            
        Returns:
            tuple: (blind_band_start, blind_band_end) where both are integers.
                Returns (None, None) if no blind band is detected.
        """
        threshold = 150
        image_filtered = torch.from_numpy(median_filter((image > threshold).to(torch.uint8),size=5))
        col_sums = torch.sum(image_filtered, dim=0)
        # Create a boolean mask for columns where the sum is exactly zero.
        zero_mask = (col_sums == 0)
        zero_indices = torch.nonzero(zero_mask, as_tuple=False).squeeze()
        
        # No blind band detected if there are no zero-sum columns.
        if zero_indices.numel() == 0:
            return None, None

        # Ensure that zero_indices is 1D.
        if zero_indices.dim() == 0:
            zero_indices = zero_indices.unsqueeze(0)
        
        # If there is only one zero column, return it as the blind band.
        if zero_indices.numel() == 1:
            blind_band_start = zero_indices.item()
            blind_band_end = blind_band_start
            return blind_band_start, blind_band_end

        # Check if all zero_indices are contiguous.
        diffs = zero_indices[1:] - zero_indices[:-1]
        if torch.all(diffs == 1):
            blind_band_start = zero_indices[0].item()
            blind_band_end = zero_indices[-1].item()
            return blind_band_start, blind_band_end
        else:
            # There are multiple contiguous blocks. Identify all blocks.
            blocks = []
            current_start = zero_indices[0].item()
            current_end = zero_indices[0].item()
            for i in range(1, len(zero_indices)):
                idx = zero_indices[i].item()
                if idx == current_end + 1:
                    current_end = idx
                else:
                    blocks.append((current_start, current_end))
                    current_start = idx
                    current_end = idx
            blocks.append((current_start, current_end))
            
            # Choose the block with maximum width.
            max_block = max(blocks, key=lambda b: b[1] - b[0] + 1)
            blind_band_start, blind_band_end = max_block
            print("Warning: Non-contiguous zero colummns detected. Using the widest contiguous block: ({}, {}). Threshold: {}".format(blind_band_start, blind_band_end, threshold))
            return blind_band_start, blind_band_end


    def adjust_patch_x(self, x, patch_width, image_width, blind_band_start, blind_band_end):
        """
        Adjust the x coordinate of a candidate patch so that it does not overlap the blind band.
        
        The candidate patch is defined by its starting x coordinate and its width.
        If the patch overlaps the blind band, we compute how much of the patch lies outside the blind zone
        on the left vs. the right, and shift the patch entirely to the side that retains more valid content.
        
        Args:
            x (int): Original x coordinate of the patch.
            patch_width (int): The width of the patch.
            image_width (int): The total width of the image.
            blind_band_start (int or None): Starting column index of the blind band.
            blind_band_end (int or None): Ending column index of the blind band.
            
        Returns:
            int or None: The adjusted x coordinate if the patch can be shifted within the image,
                        or None if it cannot be shifted without going out-of-bounds.
        """
        candidate_start = x
        candidate_end = x + patch_width - 1

        # If no blind band or candidate patch is already outside the blind band, return x unchanged.
        if blind_band_start is None or blind_band_end is None:
            return x
        if candidate_end < blind_band_start or candidate_start > blind_band_end:
            return x

        # The patch overlaps the blind band.
        # Compute valid widths on the left and right sides of the blind band.
        if candidate_start < blind_band_start:
            left_valid = min(candidate_end, blind_band_start - 1) - candidate_start + 1
        else:
            left_valid = 0
        if candidate_end > blind_band_end:
            right_valid = candidate_end - max(candidate_start, blind_band_end + 1) + 1
        else:
            right_valid = 0

        # Compute potential new x positions:
        new_x_left = blind_band_start - patch_width   # Shift so that patch ends at blind_band_start - 1.
        new_x_right = blind_band_end + 1                  # Shift so that patch starts at blind_band_end + 1.

        # Validate that the shifted patch would lie within image bounds.
        valid_left_option = (new_x_left >= 0)
        valid_right_option = (new_x_right + patch_width - 1 < image_width)

        # Decide which side to shift based on which side retains more valid content.
        if valid_left_option and valid_right_option:
            return new_x_left if left_valid >= right_valid else new_x_right
        elif valid_left_option:
            return new_x_left
        elif valid_right_option:
            return new_x_right
        else:
            return None  # Cannot adjust without going out-of-bounds.


    def extract_patches(self, image, mask, patch_size, N):
        """
        Extract N random background and composite patches while ensuring constraints on 
        foreground-background ratios, and adjust patches to avoid the blind band.
        
        Instead of skipping candidate patches that overlap the blind zone, this method 
        shifts the patch horizontally so that it lies completely outside the blind band.  
        The direction of the shift is determined by checking which side (left or right) of the 
        candidate patch contains more valid (non-blind) content.
        
        Args:
            image (Tensor): (C, H, W) Input image.
            mask (Tensor): (H, W) Binary mask (0 for background, 1 for foreground).
            patch_size (tuple): Size of patches as (height, width).
            N (int): Number of patches to extract.

        Returns:
            tuple: (foreground patches, background patches for foreground patches, 
                    duplicate background patches, foreground masks).
        """
        C, H, W = image.shape
        fg_patches, bg_patches, fg_masks = [], [], []
        fg_indices, bg_indices, bg_for_fg_patches = [], [], []
        
        
        # Get foreground and background locations.
        fg_locs = torch.nonzero(mask.squeeze())
        bg_locs = torch.nonzero(mask.squeeze() == 0)
        
        if H < patch_size[0] or W < patch_size[1]:
            return [],[],[],[]
        # Extract patches.
        for _ in range(N):  # Over-sample to account for patch rejection.
            # Sample a random location.
            y = np.random.randint(0,H - patch_size[0])
            x = np.random.randint(0,W - patch_size[1])
            # if random.random() < 0.5 and len(fg_locs) > 0:
            #     y, x = fg_locs[random.randint(0, len(fg_locs) - 1)]
            # elif len(bg_locs) > 0:
            #     y, x = bg_locs[random.randint(0, len(bg_locs) - 1)]
            # else:
            #     continue
            
            y, x = int(y), int(x)
            if y + patch_size[0] > H or x + patch_size[1] > W:
                continue

            # Adjust the x coordinate if the candidate patch overlaps the blind band.
            # new_x = self.adjust_patch_x(x, patch_size[1], W, blind_band_start, blind_band_end)
            # if new_x is None:
            #     continue  # Skip candidate if adjustment is not possible.
            # x = new_x

            patch = image[:, y:y+patch_size[0], x:x+patch_size[1]]
            patch_mask = mask[:, y:y+patch_size[0], x:x+patch_size[1]]
            
            # Skip patches that are entirely zero.
            if patch.float().mean() == 0:
                continue
            
            # Compute the foreground ratio.
            fg_ratio = patch_mask.float().mean().item()
            
            if 0.05 <= fg_ratio <= 1.0:
                fg_patches.append(patch)
                fg_masks.append(patch_mask)
                fg_indices.append(torch.tensor([y, x]))
            elif fg_ratio == 0:
                bg_patches.append(patch)
                bg_indices.append(torch.tensor([y, x]))
        
        # Convert lists to tensors if patches were successfully extracted.
        if fg_patches:
            fg_patches = torch.stack(fg_patches)
            fg_patches = self.clip(fg_patches)
            fg_patches =  TF.resize(fg_patches, self.im_size, antialias=True)
            fg_masks = TF.resize(torch.stack(fg_masks), self.im_size, antialias=True, interpolation=InterpolationMode.NEAREST)
            fg_indices = torch.stack(fg_indices)
        if bg_patches:
            bg_patches = torch.stack(bg_patches)
            bg_patches = self.clip(bg_patches)
            bg_patches = TF.resize(bg_patches, self.im_size, antialias=True)
            bg_indices = torch.stack(bg_indices)
        
        if len(fg_patches) > 0 and len(bg_patches) > 0:
            # Match nearest background patches (assuming self.find_nearest_background is defined).
            bg_for_fg_patches = self.find_nearest_background(fg_indices, bg_indices, bg_patches)
        
        return fg_patches, bg_patches, bg_for_fg_patches, fg_masks



    def _process_patches_for_testing(self, image_paths, label_paths):

        fg_images, patches_indices, masks, source_image_labels, patch_sizes, bands_info, filenames = [], [], [], [], [], [],[]
        source_counter = 1
        stride = (int(self.crop_size[0]*.1), int(self.crop_size[1]*.1))
        for img_path, lbl_path in zip(image_paths, label_paths):
            
            image = Image.open(img_path).convert("RGB")
            label = Image.open(lbl_path).convert("L")

            image = (TF.to_tensor(image)[:1, ...]* 255).to(torch.uint8)
            label = (TF.to_tensor(label)[:1, ...] * 255).to(torch.uint8)

            # Identify the blind band using the first channel of the image.
            blind_band_start, blind_band_end = self.identify_blind_band(image[0])

            if image.float().mean() == 0:
                # some images are read all zeros although they are not. Not able to solve this problem so I'm just skipping for now
                continue
            idx = 0
            for image,label in [(image[...,0:blind_band_start], label[...,0:blind_band_start]), 
            (torch.flip(image[...,blind_band_end:],dims=[-1]),torch.flip(label[...,blind_band_end:],dims=[-1]))]: 
                if image.shape[1] < self.crop_size[0] or image.shape[2] < self.crop_size[1]:
                    continue
                mult = 1 if idx == 0 else -1
                image_patches, label_patches, patch_indices_per_image, patch_sizes_per_image = self.extract_patches_with_stride(image, label, self.crop_size, stride=stride)
                fg_images.append(image_patches)
                masks.append(label_patches)
                source_image_labels.append(torch.ones(image_patches.shape[0]) * source_counter * mult)
                patch_sizes.append(patch_sizes_per_image)
                patches_indices.append(patch_indices_per_image)
                bands_info.append(torch.tensor([blind_band_start, blind_band_end]).repeat(image_patches.shape[0],1))
                filenames.extend([img_path]*len(image_patches))
                idx += 1
            source_counter += 1
        
        masks = torch.cat(masks)
        fg_images = torch.cat(fg_images)
        return fg_images, masks, torch.cat(source_image_labels), torch.cat(patch_sizes), torch.cat(patches_indices), torch.cat(bands_info), filenames

    def clip(self, images):
        mean = images.float().mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        images = images / mean
        images = torch.clip(images, 0, 16)
        return images


    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=None, collate_fn=lambda x: x,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)


class GroupedImageDataset(Dataset):
    def __init__(self, images, masks, source_image_labels, patch_sizes, patch_indices, band_info, filenames):
        """
        Args:
            images (Tensor): Tensor of image patches with shape [N, C, H, W]
            masks (Tensor): Tensor of mask patches with shape [N, 1, H, W]
            source_image_labels (Tensor): Tensor of source image labels with shape [N]
            patch_sizes (Tensor): Tensor of patch sizes with shape [N, 2]
            patch_indices (Tensor): Tensor of patch indices with shape [N, 2]
            band_info (Tensor): Tensor of band info with shape [N, 2]
        """
        self.images = images
        self.masks = masks
        self.source_image_labels = source_image_labels
        self.patch_sizes = patch_sizes
        self.patch_indices = patch_indices
        self.band_info = band_info
        self.filenames = filenames

        # Group patches by absolute value of source image label
        self.grouped_data = self.group_data_by_abs_label()

    def group_data_by_abs_label(self):
        """
        Groups the patches by the absolute value of source_image_labels.
        Returns:
            dict: Keys are the absolute values of the labels, and values are lists of indices.
        """
        grouped = {}
        for idx, label in enumerate(self.source_image_labels):
            abs_label = abs(label.item())
            if abs_label not in grouped:
                grouped[abs_label] = []
            grouped[abs_label].append(idx)
        return grouped

    def __len__(self):
        """
        Returns the number of batches (unique absolute source image labels).
        """
        return len(self.grouped_data)

    def __getitem__(self, idx):
        """
        Returns a batch of data where all patches have the same absolute source_image_label.
        """
        # Get the group of indices for a specific absolute label
        abs_label = sorted(list(self.grouped_data.keys()))[idx]
        indices = self.grouped_data[abs_label]

        # Filter the data based on the selected indices
        images_batch = self.images[indices]
        masks_batch = self.masks[indices]
        source_image_labels_batch = self.source_image_labels[indices]
        patch_sizes_batch = self.patch_sizes[indices]
        patch_indices_batch = self.patch_indices[indices]
        band_info_batch = self.band_info[indices]
        filenames = [self.filenames[i] for i in indices]
        return images_batch, masks_batch, source_image_labels_batch, patch_sizes_batch, patch_indices_batch, band_info_batch, filenames



class AugmentSSSDataModule(SSSDataModule):
    def __init__(self, data_dir, background_classifier, order_background_labels, train_val_test_split, num_workers,
                 pin_memory, crop_size, im_size, n_augments, batch_size=16, zeta=100):
        super().__init__(data_dir, background_classifier, order_background_labels, train_val_test_split, num_workers, 
        pin_memory, crop_size, im_size, batch_size, zeta)
        self.already_done = False
        self.n_augments = n_augments

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=self.im_size, scale=(0.8, 1.0)),  # Uses self.im_size
        ])

    def prepare_data(self) -> None:
        super().prepare_data()
        # Additional preparation specific to PatchBirdsDataModule if needed
        pass

    def setup(self, stage=None):
        super().setup(stage)
        if not self.already_done:
            self.data_train = self.create_augmented_dataset(torch.stack([batch[1] for batch in self.data_train]))
            self.data_val = self.create_augmented_dataset(torch.stack([batch[1] for batch in self.data_val]))
            self.data_test = self.data_val
            self.already_done = True

    def create_augmented_dataset(self, images: torch.Tensor):
        all_patches = []
        for i in range(len(images)):
            patches = []
            for j in range(self.n_augments):
                patch = self.augmentation(images[i])
                patches.append(patch)
            all_patches.append(torch.stack(patches))
        all_patches = torch.stack(all_patches)  # Shape: (num_images, n_augmentation, channels, height, width)
        return TensorDataset(all_patches)
    


    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

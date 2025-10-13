import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import wandb
import torchvision.transforms.functional as TF
from torchmetrics.functional.classification import binary_average_precision, binary_confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics.functional as TMF
from collections import defaultdict
import re
import itertools
import os
from PIL import Image

from src.models.fgbg_module import FgBgLitModule
from src.utils.utils import normalize_images
from src.utils.utils import apply_colormap_to_tensor


class SSSLightningModule(FgBgLitModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            num_projections: int,
            max_iter_dswd: int,
            sw_encoder,
            sw_encoder_checkpoint: str,
            regularization_lambda_dswd: float,
            image_size,
            input_channels: int,
            embedding_dim: float,
            original_crop_size,
            divergence_loss='dswd',
            cirtic_task: str = None,
            psudo_colormap: str = None,
            images_to_plot_num: int = 32
    ):
        self.original_crop_size = original_crop_size
        super().__init__(net, optimizer, scheduler, compile, num_projections, max_iter_dswd, sw_encoder,
                         sw_encoder_checkpoint,
                         regularization_lambda_dswd, image_size, input_channels, embedding_dim, 1, divergence_loss,'fid',
                         cirtic_task,
                         psudo_colormap,
                         images_to_plot_num)

    def stitch_and_visualize(self, images, masks, source_labels, patches_dims, estimated_masks, pseudo_colormap):
        """
        Stitches non-overlapping patches of images and masks based on source_labels and patches_dims, and visualizes them.

        Args:
            images (Tensor): (N, C, H, W) Batched input images.
            masks (Tensor): (N, 1, H, W) Ground truth masks.
            source_labels (Tensor): (N,) Tensor containing source image identifiers.
            patches_dims (Tensor): (N, 2) Tensor containing (rows, cols) for each source image.
            estimated_masks (Tensor): (N, 1, H, W) Model's estimated masks.
            pseudo_colormap (str): Colormap for visualizing input images.

        Returns:
            figures (list): List of Matplotlib figures, each representing a unique source.
        """

        unique_sources = source_labels.unique()
        figures = []

        for source in unique_sources:
            indices = (source_labels == source).nonzero(as_tuple=True)[0]

            # Get patch size
            _, C, patch_H, patch_W = images.shape

            # Extract grid size for this source
            source_idx = indices[0]  # All patches from the same source share the same (rows, cols)
            grid_rows, grid_cols = patches_dims[source_idx].tolist()

            # Create blank canvases for reconstruction
            stitched_img = torch.zeros((C, grid_rows * patch_H, grid_cols * patch_W))
            stitched_gt = torch.zeros((grid_rows * patch_H, grid_cols * patch_W))
            stitched_est = torch.zeros((grid_rows * patch_H, grid_cols * patch_W))

            # Arrange patches based on known (row, col) structure
            for i, idx in enumerate(indices):
                row, col = divmod(i, grid_cols)
                y_start, x_start = row * patch_H, col * patch_W

                stitched_img[:, y_start:y_start + patch_H, x_start:x_start + patch_W] = images[idx]
                stitched_gt[y_start:y_start + patch_H, x_start:x_start + patch_W] = masks[idx].squeeze()
                stitched_est[y_start:y_start + patch_H, x_start:x_start + patch_W] = estimated_masks[idx].squeeze()

            stitched_est = stitched_est * (stitched_img > 1e-4).to(stitched_est)
            stitched_thresh = (stitched_est > 0.5).float()
            # Create figure for this source
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))

            # Plot input image
            axes[0].imshow(stitched_img.permute(1, 2, 0).cpu().numpy(), cmap=pseudo_colormap)
            axes[0].axis("off")

            # Plot ground truth mask
            axes[1].imshow(stitched_gt.cpu().numpy(), cmap="gray")
            axes[1].axis("off")

            # Plot estimated mask
            axes[2].imshow(stitched_est.cpu().squeeze().numpy(), cmap="gray")
            axes[2].axis("off")

            # Plot thresholded estimated mask
            axes[3].imshow(stitched_thresh.cpu().squeeze().numpy(), cmap="gray")
            axes[3].axis("off")

            plt.tight_layout()
            figures.append(fig)
            plt.close()

        return figures


    def create_dataset(self, dataloader):
        """Create a dataset of (image, label) pairs"""
        fp_samples, tp_samples = [], []

        for batch in dataloader:
            fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, _ = batch

            fg_images = normalize_images(fg_images).to(self.device)
            bg_images = normalize_images(bg_images).to(self.device)
            with torch.inference_mode():
                with torch.no_grad():
                    bg_estimated_masks, fg_estimated_masks, _, _= self.pass_to_mask_net(bg_images, fg_images, 'after_train')

            # Identify FP (background misclassified as foreground)
            fp_mask = bg_estimated_masks.mean(dim=(1, 2, 3)) > 0.1
            fp_samples.extend(bg_images[fp_mask].cpu())

            # TP samples are the actual foreground images
            tp_samples.extend(fg_images.cpu())

        # Convert to tensor dataset
        X = torch.stack(fp_samples + tp_samples)
        y = torch.tensor([0] * len(fp_samples) + [1] * len(tp_samples), dtype=torch.float32)
        return TensorDataset(X, y)

    @torch.enable_grad()
    @torch.inference_mode(mode=False)
    def train_resnet(self, train_loader, val_loader):
        """Train ResNet-18 to classify FP vs. TP"""
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Binary classification
        
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1 channel
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None
        )

        optimizer = optim.Adam(self.resnet.parameters(), lr=1e-4)
        self.resnet.to(self.device)

        for epoch in range(20):  # Number of epochs
            self.resnet.train()
            epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.resnet(images).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.sigmoid(outputs) > 0.5
                epoch_loss += loss.item()
                epoch_acc += TMF.accuracy(preds, labels, task="binary").item()
                epoch_f1 += TMF.f1_score(preds, labels, task="binary").item()

            # Validation Step
            self.resnet.eval()
            val_loss, val_acc, val_f1 = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.resnet(images).squeeze(1)
                    loss = F.binary_cross_entropy_with_logits(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                    val_loss += loss.item()
                    val_acc += TMF.accuracy(preds, labels, task="binary").item()
                    val_f1 += TMF.f1_score(preds, labels, task="binary").item()

            # Print epoch results
            print(f"Epoch {epoch+1}: Train Loss={epoch_loss/len(train_loader):.4f}, Train Acc={epoch_acc/len(train_loader):.4f}, Train F1={epoch_f1/len(train_loader):.4f}")
            print(f"            Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc/len(val_loader):.4f}, Val F1={val_f1/len(val_loader):.4f}")


    def on_test_start(self):

        """Train ResNet-18 to distinguish FP vs. TP after training the main model"""
        # print("Extracting FP and TP samples...")

        # train_dataset = self.create_dataset(self.trainer.datamodule.train_dataloader())
        # val_dataset = self.create_dataset( self.trainer.datamodule.val_dataloader())

        # # Split into train and validation loaders
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # print("Training ResNet-18 for FP vs. TP classification...")
        # self.train_resnet(train_loader, val_loader)

        self.estimated_masks = []
        self.gt = []
        self.filenames = []
        self.patch_stitcher = PatchStitcher(self.original_crop_size,
                                            colormap=self.psudo_colormap)  # Initialize with desired colormap

    def test_step(self, batch, batch_idx: int) -> None:
        images, masks, source_labels, patch_sizes, patches_indices, band_info, filenames = batch

        images = normalize_images(images)

        batch_size = 128  # 
        estimated_masks_list = []
        # Process images in smaller batches
        for i in range(0, images.shape[0], batch_size):
            batch_images = images[i:i + batch_size]
            logits = self.forward(batch_images.float())
            batch_estimated_masks = torch.sigmoid(logits)

            # potential_fg = batch_estimated_masks.mean((1,2,3)) > .1
            # tps = torch.sigmoid(self.resnet(batch_images[potential_fg]))
            # fps = tps < .5
            # if fps.sum() > 1:
            #     batch_estimated_masks[potential_fg][fps] = torch.zeros_like(batch_estimated_masks[potential_fg][fps])
            estimated_masks_list.append(batch_estimated_masks)

        # Concatenate all estimated masks
        estimated_masks = torch.cat(estimated_masks_list, dim=0)

        # Stitch patches using full dataset
        fig = self.patch_stitcher.stitch_patches(images.cpu(), masks.cpu(), estimated_masks.cpu(),
                                                patches_indices.cpu(), patch_sizes.cpu(), source_labels.cpu(),
                                                band_info.cpu(), filenames)
        if fig is None:
            return

        logger = self.logger.experiment
        if hasattr(logger, 'add_image'):
            logger.add_image(f'test/segmentation', fig)
        else:
            logger.log({f'test/segmentation': wandb.Image(fig)})

        # Store results for further evaluation
        self.estimated_masks.append(estimated_masks.to(torch.float16).cpu())
        self.gt.append(masks.to(torch.bool).cpu())
        filenames = [os.path.basename(filename) for filename in filenames]
        self.filenames.append(filenames)


    def group_filenames_by_site(self, filenames):
        site_dict = defaultdict(list)
        
        for index, filename in enumerate(filenames):
            match = re.match(r"(.+)_\d+\.png", filename)
            if match:
                site_name = match.group(1)
                site_dict[site_name].append(index)
        
        return dict(site_dict)


    def compute_metrics(self, estimated_masks, gt):

        estimated_masks = estimated_masks.flatten()
        gt = gt.flatten()
        # Compute AP and Mean IoU
        ap = binary_average_precision(estimated_masks, gt)
        
        # Compute Confusion Matrix
        conf = binary_confusion_matrix(estimated_masks > 0.5, gt)
        
        # Extract TP, TN, FP, FN
        tn, fp = conf[0, 0], conf[0, 1]
        fn, tp = conf[1, 0], conf[1, 1]
        
        # Compute Additional Metrics
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        meanIoU = tp / (tp + fn + fp + 1e-8)
        
        return {"AP": ap, "F1": f1_score, "meanIoU": meanIoU}

    def compute_metrics_by_site(self, estimated_masks, gt, filenames):
        site_indices = self.group_filenames_by_site(filenames)
        site_metrics = {}
        
        for site, indices in site_indices.items():
            site_estimated = estimated_masks[indices]
            site_gt = gt[indices]
            
            site_metrics[site] = self.compute_metrics(site_estimated, site_gt)
            
            # Log Metrics
            for metric, value in site_metrics[site].items():
                self.log(f"{site}_{metric}", value)
        
        return site_metrics

    def on_test_epoch_end(self):
        estimated_masks = torch.cat(self.estimated_masks)
        gt = torch.cat(self.gt)
        filenames = list(itertools.chain(*self.filenames)) 
        overall_metrics =self.compute_metrics(estimated_masks, gt)
        
        # Log overall metrics
        for metric, value in overall_metrics.items():
            self.log(metric, value)
        
        micro_ap = []
        micro_f1 = []
        micro_meanIOU = []
        num_images = len(estimated_masks)
        for index in range(num_images):
            metric = self.compute_metrics(estimated_masks[index],gt[index])
            micro_ap.append(metric['AP'])
            micro_f1.append(metric['F1'])
            micro_meanIOU.append(metric['meanIoU'])

        self.log('mean_AP',np.mean(micro_ap))
        self.log('mean_F1',np.mean(micro_f1))
        self.log('mean_meanIOU',np.mean(micro_meanIOU))
        self.log('std_AP',np.std(micro_ap))
        self.log('std_F1',np.std(micro_f1))
        self.log('std_meanIOU',np.std(micro_meanIOU))


        # Compute and log per-site metrics
        self.compute_metrics_by_site(estimated_masks, gt, filenames)



class PatchStitcher:
    def __init__(self, original_crop_size, colormap):
        self.pusodo_colormap = colormap
        self.original_crop_size = original_crop_size

    def calculate_image_size(self, patch_indices, patch_sizes):
        """
        Calculate the original image size from patch indices and patch sizes.
        Args:
            patch_indices (Tensor): Indices for each patch's start position.
            patch_sizes (Tensor): Sizes of each patch.
        Returns:
            tuple: Original image size (height, width).
        """
        max_h = torch.max(patch_indices[:, 0] + patch_sizes[:, 0])  # Calculate max height
        max_w = torch.max(patch_indices[:, 1] + patch_sizes[:, 1])  # Calculate max width

        min_h = torch.min(patch_indices[:, 0])  # Calculate min height
        min_w = torch.min(patch_indices[:, 1])  # Calculate min width

        return (max_h - min_h).item(), (max_w - min_w).item()

    def merge_overlapping_patches(self, image_patches, mask_patches, estimated_mask_patches, patch_indices, patch_sizes,
                                  h, w):
        """
        Merge overlapping patches by averaging the overlapping region.
        Args:
            image_patches (Tensor): Image patches.
            mask_patches (Tensor): Mask patches.
            estimated_mask_patches (Tensor): Estimated mask patches.
            patch_indices (Tensor): Indices for each patch's start position.
            patch_sizes (Tensor): Sizes of each patch.
            h (int): Height of the original image.
            w (int): Width of the original image.
        Returns:
            Tensor: Merged image.
            Tensor: Merged ground truth mask.
            Tensor: Merged estimated mask.
        """
        _, c, _, _ = image_patches.shape
        stitched_image = torch.zeros(c, h, w)
        stitched_mask = torch.zeros(c, h, w)
        stitched_estimated_mask = torch.zeros(c, h, w)

        count = torch.zeros(c, h, w)  # To count overlaps

        for idx, (patch_start, patch_size) in enumerate(zip(patch_indices, patch_sizes)):
            start_h, start_w = patch_start
            patch_h, patch_w = patch_size

            # Calculate the region to update in the stitched image
            stitched_image[:, start_h:start_h + patch_h, start_w:start_w + patch_w] += image_patches[idx]
            stitched_mask[:, start_h:start_h + patch_h, start_w:start_w + patch_w] += mask_patches[idx]
            stitched_estimated_mask[:, start_h:start_h + patch_h, start_w:start_w + patch_w] += estimated_mask_patches[
                idx]

            count[:, start_h:start_h + patch_h, start_w:start_w + patch_w] += 1

        # Normalize by the number of overlaps (taking the mean)
        stitched_image /= count
        stitched_mask /= count
        stitched_estimated_mask /= count

        return stitched_image, stitched_mask, stitched_estimated_mask

    def flip_right_side(self, image, mask, estimated_mask):
        """
        Flip the right side (negative label) 180 degrees clockwise.
        Args:
            image (Tensor): Image tensor.
            mask (Tensor): Ground truth mask tensor.
            estimated_mask (Tensor): Estimated mask tensor.
        Returns:
            Tensor: Flipped image.
            Tensor: Flipped mask.
            Tensor: Flipped estimated mask.
        """
        # Flip along the horizontal axis (width)
        image_flipped = torch.flip(image, dims=[2])
        mask_flipped = torch.flip(mask, dims=[2])
        estimated_mask_flipped = torch.flip(estimated_mask, dims=[2])

        return image_flipped, mask_flipped, estimated_mask_flipped

    def create_zero_zone(self, band_info, h, w):
        """
        Create a zero zone using the band information.
        Args:
            band_info (Tensor): Information about where the zero band starts and ends.
            h (int): Height of the original image.
            w (int): Width of the original image.
        Returns:
            Tensor: Zero zone tensor.
        """
        zero_zone = torch.zeros(1, h, w)
        for start, end in band_info:
            zero_zone[0, :, start:end] = 1  # Marking the zero zone with 1s
        return zero_zone

    def make_test_image_grid(self,img_path,full_mask,full_estimated_mask):
        # Function to load an image as a tensor
        def load_image(img_path):
            image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
            transform = torchvision.transforms.ToTensor()  # Convert to (C, H, W) format
            return transform(image)

        # Load and process the image
        full_image = load_image(img_path).to(full_mask)[:1].unsqueeze(0)  # Load the image as a tensor
        full_image = apply_colormap_to_tensor(full_image)[0]  # Apply pseudo colormap


        resize_nn = torchvision.transforms.Resize(full_image.shape[1:], interpolation=torchvision.transforms.InterpolationMode.NEAREST)  # Nearest-neighbor for binary mask
        resize_bilinear = torchvision.transforms.Resize(full_image.shape[1:], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)  # Bilinear for soft masks

        # Resize masks
        full_mask = resize_nn(full_mask)  # Binary mask (nearest neighbor)
        full_estimated_mask = resize_bilinear(full_estimated_mask)
        
        # Normalize grayscale images before visualization
        def preprocess_for_visualization(img):
            if img.shape[0] == 1:  # Convert grayscale (1, H, W) to RGB (3, H, W)
                img = img.repeat(3, 1, 1)
            return img

        # Prepare images for visualization
        images_to_visualize = [
            full_image,  # Original image with colormap
            preprocess_for_visualization(full_mask),  # Ground truth mask
            preprocess_for_visualization(full_estimated_mask),  # Estimated mask
            preprocess_for_visualization((full_estimated_mask > 0.5).float()),  # Thresholded mask
        ]

        # Create a grid of images
        grid = torchvision.utils.make_grid(images_to_visualize, nrow=4, padding=5, normalize=True, pad_value=1)
        return grid




    def stitch_patches(self, patch_images, patch_masks, patch_estimated_masks, patch_indices, patch_sizes,
                       source_labels, band_info, img_path):
        """
        Stitch patches together to form the complete image, GT mask, and estimated mask.
        Args:
            patch_images (Tensor): Image patches.
            patch_masks (Tensor): Mask patches.
            patch_estimated_masks (Tensor): Estimated mask patches.
            patch_indices (Tensor): Indices for each patch's start position.
            patch_sizes (Tensor): Sizes of each patch.
            source_labels (Tensor): Labels (positive for left, negative for right).
            band_info (Tensor): Information about where the zero band starts and ends.
        Returns:
            fig: A matplotlib figure containing the full image, GT mask, and estimated mask.
        """
        patch_images = TF.resize(patch_images, self.original_crop_size, antialias=True)
        patch_estimated_masks = TF.resize(patch_estimated_masks, self.original_crop_size, antialias=True)
        patch_masks = TF.resize(patch_masks, self.original_crop_size,
                                interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True)
        # Separate the patches based on the source labels
        left_image_patches = patch_images[source_labels > 0]
        left_mask_patches = patch_masks[source_labels > 0]
        left_estimated_mask_patches = patch_estimated_masks[source_labels > 0]
        left_patch_sizes = patch_sizes[source_labels > 0]
        left_patch_indices = patch_indices[source_labels > 0]

        right_image_patches = patch_images[source_labels < 0]
        right_mask_patches = patch_masks[source_labels < 0]
        right_estimated_mask_patches = patch_estimated_masks[source_labels < 0]
        right_patch_sizes = patch_sizes[source_labels < 0]
        right_patch_indices = patch_indices[source_labels < 0]

        if len(left_image_patches) == 0 or len(right_image_patches) == 0:
            return None

        left_h, left_w = self.calculate_image_size(left_patch_indices, left_patch_sizes)
        right_h, right_w = self.calculate_image_size(right_patch_indices, right_patch_sizes)

        assert left_h == right_h, "this algorithm assumes that the band spans over the height. "
        # Step 2: Merge overlapping patches for both left and right sides
        stitched_left_image, stitched_left_mask, stitched_left_estimated_mask = self.merge_overlapping_patches(
            left_image_patches, left_mask_patches, left_estimated_mask_patches, left_patch_indices, left_patch_sizes,
            left_h, left_w)

        stitched_right_image, stitched_right_mask, stitched_right_estimated_mask = self.merge_overlapping_patches(
            right_image_patches, right_mask_patches, right_estimated_mask_patches, right_patch_indices,
            right_patch_sizes, right_h, right_w)

        # Step 3: Flip the right side 180 degrees
        flipped_right_image, flipped_right_mask, flipped_right_estimated_mask = self.flip_right_side(
            stitched_right_image, stitched_right_mask, stitched_right_estimated_mask)

        # Step 4: Create zero zone
        zero_zone = torch.zeros(patch_images.shape[1], left_h, band_info[0, 1] - band_info[0, 0])

        # Step 6: Stitch the left and right parts together with the zero zone in between
        full_image = torch.cat((stitched_left_image, zero_zone, flipped_right_image), dim=2)
        full_mask = torch.cat((stitched_left_mask, zero_zone, flipped_right_mask), dim=2)
        full_estimated_mask = torch.cat((stitched_left_estimated_mask, zero_zone, flipped_right_estimated_mask), dim=2)

        # band_area = torch.zeros_like(full_image)
        # band_area[...,band_info[0,0]:band_info[0,1]] = 1
        # # Step 7: Visualizing images
        # fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        # #read img_path and use pusodo_colormap and call it full_image
        # axes[0].imshow(full_image.cpu().numpy().transpose(1, 2, 0), cmap=self.pusodo_colormap)
        # axes[1].imshow(full_mask.cpu().numpy().squeeze(), cmap="gray")
        # axes[2].imshow(full_estimated_mask.cpu().numpy().squeeze(), cmap="gray")
        # axes[3].imshow(full_estimated_mask.cpu().numpy().squeeze() > 0.5, cmap="gray")
        # axes[4].imshow(band_area.cpu().numpy().squeeze(), cmap="gray")
        # axes[4].imshow(full_image.cpu().numpy().transpose(1, 2, 0), cmap=self.pusodo_colormap, alpha=.4)

        # for ax in axes:
        #     ax.axis("off")
        # plt.close()
        grid = self.make_test_image_grid(img_path[0],full_mask,full_estimated_mask)
        return grid

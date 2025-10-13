from typing import List
import segmentation_models_pytorch as smp
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from dice_puma import compute_dice_scores, calculate_micro_dice_score_with_masks
import os
import tifffile
import torch
from PIL import Image
import kornia.augmentation as K
import kornia.geometry.transform as T
import kornia.morphology as KM
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import shutil
from typing import List
import json
from shapely.geometry import shape
from rasterio.features import rasterize
def Mine_resize(image = None, mask = None, final_size = None):
    """Resize image and mask to the final size."""
    image_resized = F.interpolate(image, size=(final_size[0], final_size[1]), mode="bilinear",
                                  align_corners=False)
    mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=final_size, mode="nearest").squeeze(1).long()
    return image_resized, mask_resized






class RandomOneOf(torch.nn.Module):
    def __init__(self, augmentations, p=1.0):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x  # No augmentation applied
        # Randomly select one augmentation
        idx = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[idx](x)


class KorniaAugmentation:
    def __init__(self, mode="train", num_classes=6, seed=None, size = None):
        self.mode = mode
        self.size = size
        self.num_classes = num_classes
        self.seed = seed
        torch.manual_seed(seed) if seed else None

        # Define PiecewiseAffine augmentation
        self.piecewise_affine = K.RandomThinPlateSpline(scale=0.1, align_corners=False)

        # Define Shape Transformations
        # self.affine = T.Compose([
        #     T.RandomAffine(
        #         degrees=(-179, 179),  # Rotation between -179 and 179 degrees
        #         translate=(0.01, 0.01),  # Translation up to 1% of image dimensions
        #         scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
        #         shear=(-5, 5),  # Shear range between -5 and 5 degrees
        #         # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
        #         fill=0  # Fill padding areas with black (0)
        #     )
        # ])

        # Create a RandomOneOf augmentation for the shape transformations
        self.shape_augs = torch.nn.Sequential(#RandomOneOf([
            # K.RandomThinPlateSpline(p=0.5, scale=0.1, same_on_batch=True, keepdim=True),  # Correct parameters
            T.RandomAffine(
                degrees=(-179, 179),  # Rotation between -179 and 179 degrees
                translate=(0.01, 0.01),  # Translation up to 1% of image dimensions
                scale=(1, 1),  # Scaling between 0.8 and 1.2
                shear=(-5, 5),  # Shear range between -5 and 5 degrees
                # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
                fill=0  # Fill padding areas with black (0)
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            # K.CenterCrop(size=size)
        )

        # Define Input Transformations
        self.input_augs = torch.nn.Sequential(
            RandomOneOf([
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                K.RandomMedianBlur((3, 3), p=1.0),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
            ], p=1.0),
            K.RandomBrightness(brightness=(0.9, 1.1), p=1.0),
            K.RandomContrast(contrast=(0.75, 1.25), p=1.0),
            K.RandomHue(hue=(-0.05, 0.05), p=1.0),
            K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),

        )

    def __apply_piecewise_affine(self, images, masks):
        """Apply piecewise affine transformation to both images and masks."""
        # Stack masks into additional channels

        B, C, H, W = images.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Create a temporary ones tensor to track padding
        ones_tensor = torch.ones_like(images)

        # Combine images, masks, and the ones tensor for padding tracking
        combined = torch.cat([images, masks_one_hot, ones_tensor], dim=1)

        # Apply affine transformation to the combined tensor (image + mask + ones)
        combined_augmented = self.piecewise_affine(combined)

        # Separate images, masks, and ones tensor after transformation
        images_aug = combined_augmented[:, :C]
        masks_aug = combined_augmented[:, C:C + self.num_classes]
        ones_aug = combined_augmented[:, C + self.num_classes:]


        masks_aug = masks_aug.argmax(dim=1).long()  # Convert back to class labels

        # Track padding by checking where ones_aug has been turned to 0 (padding areas)
        padding_mask = ones_aug == 0  # This will be True in the padding areas

        # Fill padding for images with 1 (where padding_mask is True)
        images_aug = torch.where(padding_mask, torch.tensor(1.0, device=images_aug.device), images_aug)
        return images_aug, masks_aug

    def __apply_erode_margins(self, masks):
        """Add margins to masks by applying erosion."""
        B, H, W = masks.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply erosion to each class
        kernel = torch.ones((3, 3), device=masks.device)
        eroded_channels = []
        for i in range(self.num_classes):
            mask_channel = masks_one_hot[:, i:i + 1]
            eroded = mask_channel
            for _ in range(5):  # Perform erosion 3 times
                eroded = KM.erosion(eroded, kernel)
            eroded_channels.append(eroded)

        # Combine eroded masks
        eroded_masks = torch.cat(eroded_channels, dim=1)
        combined_masks = eroded_masks.argmax(dim=1).long()

        return combined_masks


    def __call__(self, image, mask):
        # Step 1: Apply margins to masks
        # mask = self.__apply_erode_margins(mask)
        # Step 4: Apply input augmentations
        image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])

        # # Step 2: Apply piecewise affine transformation
        # image, mask = self.__apply_piecewise_affine(image, mask)

        # Step 3: Apply shape augmentations using RandomOneOf
        # Add the temporary tensor of ones to track padding
        ones_tensor = torch.ones_like(image)

        combined = torch.cat([image, mask.unsqueeze(1).float(), ones_tensor], dim=1)

        # Apply the augmentations to the combined tensor (image + mask + ones)
        combined_augmented = self.shape_augs(combined)

        # After augmentation, split the tensor back into the image, the mask, and the ones tensor
        image = combined_augmented[:, :image.size(1)]  # Only the image part
        mask = combined_augmented[:, image.size(1):image.size(1) + mask.unsqueeze(1).size(1)]  # Only the mask part
        ones_aug = combined_augmented[:, image.size(1) + mask.size(1):]  # Only the ones tensor

        # Set padding areas of the image to 1 (based on ones_aug tracking)
        padding_mask = ones_aug == 0  # This will be True in padding areas
        image = torch.where(padding_mask, torch.tensor(1.0, device=image.device), image)
        #
        # # Step 4: Apply input augmentations
        # image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])
        #
        # # Step 5: Combine augmented mask with the image
        # # image1 = (1 - mask) + image
        #
        # # Step 6: Resize to final size
        mask = mask.squeeze(1)
        return image, mask


def Data_class_analyze(masks,# a numpy array (batch, H, W, C)
                       class_labels): # A list of classes with correct order. for example class0 should be placed in first place.
    """This function analyzes the images and masks. it counts number and area of samples for each class.
    It is a good representation for imbalance data. It also gives hints to data augmentation and over-sampling"""

    shape = np.shape(masks)
    num_classes = len(class_labels)
    class_samples = np.zeros(num_classes)
    class_areas = np.zeros(num_classes)
    class_distribution = np.zeros((shape[0],num_classes))
    for i in range(shape[0]):
        msk = masks[i]
        if len(msk.shape)>2:
            msk = np.sum(msk, axis=2)# convert to one image
        for j in range(num_classes):
            area = np.sum(msk == j)
            if area > 0:
                # if j == 5:
                #     plt.imshow(msk)
                #     plt.show()
                class_samples[j] += 1
                class_areas[j] += area
                class_distribution[i,j] += 1
    return class_samples, class_areas, class_distribution



def split_train_val(class_samples = 0, class_distribution = 0, val_percent = 0.2):
    a = 0
    class_distribution1 = np.copy(class_distribution)
    shape = np.shape(class_distribution)
    val_samples = int(shape[0]*val_percent)
    train_samples = shape[0] - val_samples
    val_index = np.zeros(val_samples)
    train_index = np.zeros(train_samples)
    val_indexes = 0
    val_samples1 = np.copy(val_samples)
    for i in range(1,shape[1]):
        ind = np.where(class_samples[1:] == np.min(class_samples[1:]))[0]+1
        ind = ind[0]
        val = class_samples[ind]
        class_samples[ind] = np.inf
        random.seed(42)
        val_sample_inds = random.sample(range(int(val)), int(np.ceil(val*val_percent)))
        print(val_sample_inds)
        k = 0
        for m in range(shape[0]):
            if class_distribution[m,int(ind)] == 1:
                if len(np.where(np.array(val_sample_inds) == k)[0]):
                    val_index[val_indexes] = m
                    class_distribution[m] = 0*class_distribution[m]
                    val_indexes += 1
                    val_samples1 -= 1
                    if val_samples1 == 0:
                        break
                    # print(k)
                k += 1
        if val_samples1 ==0:
            break
    val_index = np.where(np.sum(class_distribution,axis = 1) == 0)
    train_index = np.where(np.sum(class_distribution, axis=1) != 0)

    # print(val_index)





    return train_index, val_index


def addsamples(images,mask,sample_th = 0.2, tissue_labels = None):
    a = 2
    angles = [45,90,135,180,225,270,325]
    shape = np.shape(images)
    shape_m = np.shape(mask)
    new_ims = []
    new_masks = []
    class_samples, class_areas, class_distribution = Data_class_analyze(mask, tissue_labels)
    average_class = class_samples / np.shape(mask)[0]
    average_area = class_areas / np.sum(class_areas)
    avg = 100 * average_class * 100 * average_area / np.sum(100 * average_class * 100 * average_area)
    new_ims = []
    new_masks = []
    for i in range(1,shape_m[3]):# start from 1 to skip background
        ind = np.where(avg[1:] == np.min(avg[1:]))[0]+1
        ind = ind[0]
        val = avg[ind]
        avg[ind] = np.inf
        class_inds = np.where(class_distribution[:,ind] == 1)
        if val < sample_th:
            up_num = int(shape[0] * sample_th/(1-sample_th) - 100*average_area[ind])
            for j in range(3):# fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break
                    new_ims.append(cv2.flip(images[inds],j-1))
                    new_masks.append(cv2.flip(mask[inds],j-1))
                    up_num -= 1

            for rot in angles:
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                    imm = cv2.warpAffine(images[inds], M, (shape[1], shape[2]),borderValue=(255,255,255))
                    new_ims.append(imm)
                    imm = cv2.warpAffine(np.array(mask[inds], dtype=np.uint8), M, (shape[1], shape[2]))
                    new_masks.append(imm)


                    up_num -= 1

            for j in range(3):  # fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    image = cv2.flip(images[inds], j-1)
                    msk = cv2.flip(mask[inds], j-1)
                    for rot in angles:
                        if up_num <= 0:
                            break
                        M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                        imm = cv2.warpAffine(image, M, (shape[1], shape[2]),borderValue=(255,255,255))
                        new_ims.append(imm)
                        imm = cv2.warpAffine(np.array(msk, dtype=np.uint8), M, (shape[1], shape[2]))
                        new_masks.append(imm)
                        up_num -= 1


            class_distribution[class_inds] = 0 * class_distribution[class_inds]
    a = 0
    images1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape[3]))
    mask1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape_m[3]))
    images1[0:shape[0]] = images
    mask1[0:shape[0]] = mask

    new_ims1 = np.array(new_ims)
    new_masks1 = np.array(new_masks, dtype=np.uint8)
    images1[shape[0]:] = new_ims1
    mask1[shape[0]:] = new_masks1
    images1 = images1.astype('float32')
    mask1 = mask1.astype('int64')

    return images1, mask1




def puma_dice_loss(preds, targets, eps=1e-6):
    """
    Compute the Dice loss for binary or multi-class segmentation.

    Args:
        preds (torch.Tensor): Predicted tensor of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth tensor of shape (B, H, W).
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss value.
    """
    num_classes = preds.size(1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
    union = torch.sum(preds + targets_one_hot, dim=(2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()  # Dice loss is 1 - Dice score


# def compute_puma_dice_micro_dice(model = None, target_siz = None,epoch = 1, input_folder = '', output_folder = '', ground_truth_folder = '', device = None, model1=None, weights_list = None):
#     # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
#     # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     if epoch == 1:
#         if os.path.exists(output_folder):
#             for root, dirs, files in os.walk(output_folder, topdown=False):
#                 for file in files:
#                     os.remove(os.path.join(root, file))
#                 for dir in dirs:
#                     os.rmdir(os.path.join(root, dir))
#             os.rmdir(output_folder)
#         os.makedirs(output_folder, exist_ok=True)
#
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Define preprocessing transforms
#
#     # Process each image
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith(".tif"):
#             input_path = os.path.join(input_folder, file_name)
#             output_path = os.path.join(output_folder, file_name)
#
#             # Read the TIF image
#             image = tifffile.imread(input_path)
#
#             # Ensure the image has 3 channels
#             # Handle 4-channel images by dropping the alpha channel
#             if image.shape[2] == 4:
#                 image = image[:, :, :3]  # Keep only the first three channels (RGB)
#             elif image.shape[2] != 3:
#                 raise ValueError(f"Unexpected number of channels in image: {file_name}")
#             image = cv2.resize(image, target_siz)
#
#             image = image / 255
#
#             image = np.transpose(image, (2, 0, 1))
#             image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
#
#             # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#             # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
#             # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]
#
#             # Get prediction
#             if weights_list != None:
#                 for k in range(len(weights_list)):
#                     model.load_state_dict(torch.load(weights_list[k], weights_only=True))
#                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                     model.to(device)
#                     model.eval()
#                     if k == 0:
#                         prediction = F.softmax(model(image_tensor), dim=1)
#
#                     else:
#                         prediction = prediction + F.softmax(model(image_tensor), dim=1)
#                 prediction = prediction / len(weights_list)
#                 # prediction = F.softmax(prediction, dim=1)
#             else:
#                 prediction = model(image_tensor)
#                 prediction = F.softmax(prediction, dim=1)
#                 if model1 is not None:
#                     prediction1 = model1(image_tensor)
#                     prediction1 = F.softmax(prediction1, dim=1)
#
#                     prediction = 0.5*prediction + 0.5*prediction1
#
#                 # Post-process prediction (e.g., apply softmax or argmax)
#             prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()
#             prediction[prediction>5] = prediction[prediction>5]
#
#             # Save the prediction as a TIF file
#             with tifffile.TiffWriter(output_path) as tif:
#                 tif.write(prediction.astype(np.uint8), resolution=(300, 300))
#
#             # print(f"Processed and saved: {file_name}")
#     # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
#     prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
#     image_shape = (1024, 1024)  # Adjust based on your images' resolution
#
#     dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
#                                                                          image_shape)
#     # print("Overall Dice Scores:", mean_puma_dice)
#     # print("Overall Mean Dice Scores:", mean_dice_classes)
#
#     # for file, scores in dice_scores.items():
#     #     print(f"{file}: {scores}")
#
#
#     micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
#                                                                          image_shape, eps=0.00001)
#
#
#     return mean_puma_dice, micro_dices, mean_micro_dice


def circular_augmentation(train_images, masks, target_class, r1, r2, d1):
    """
    Apply circular augmentation to the specified class in the segmentation mask.

    Parameters:
        train_images (torch.Tensor): Input tensor of training images, size (B, C, H, W).
        masks (torch.Tensor): Input tensor of mask images, size (B, H, W) (torch.long).
        target_class (int): The class on which to apply the augmentation.
        r1 (int): Minimum radius of circles.
        r2 (int): Maximum radius of circles.
        d1 (float): Density of circles (fraction of target class area to be covered).

    Returns:
        tuple: Augmented training images and masks.
    """
    # Get device from input tensors
    device = train_images.device

    # Get dimensions
    B, C, H, W = train_images.shape
    augmented_images = train_images.clone()
    augmented_masks = masks.clone()

    for b in range(B):
        # Extract the target class region from the mask
        target_region = (masks[b] == target_class)
        target_area = target_region.sum().item()

        if target_area == 0:
            continue  # Skip if the target class is not present

        # Calculate maximum allowable circle area
        max_circle_area = target_area * d1
        current_area = 0

        circles = []  # Track placed circles as (x, y, r)

        while current_area < max_circle_area:
            # Random radius
            r = np.random.randint(r1, r2 + 1)

            # Generate a random center within the valid target region
            valid_y, valid_x = torch.where(target_region)
            if len(valid_y) == 0:
                break  # Exit if no valid points are left
            idx = np.random.randint(len(valid_y))
            x, y = valid_x[idx].item(), valid_y[idx].item()

            # Check for overlap
            overlap = False
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < r + cr:
                    overlap = True
                    break

            if not overlap:
                # Generate a circular area
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                circle_area = ((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2

                # Ensure the circle is within the target region
                circle_mask = circle_area & target_region

                new_area = circle_mask.sum().item()
                # if current_area + new_area <= max_circle_area:
                    # Apply the augmentation: set the mask to 0 and the image to 0 in the circle
                augmented_masks[b][circle_mask] = 0
                augmented_images[b][:, circle_mask] = 1

                current_area += new_area
                circles.append((x, y, r))

    return augmented_images, augmented_masks

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def upsample_necro(image_data = None,mask_data = None):
    rows, cols = np.shape(image_data)[1], np.shape(image_data)[2]  # Replace with your image dimensions
    r, c = np.indices((rows, cols))

    # Create masks
    upper_diag = c >= r
    lower_diag = c <= r
    upper_anti_diag = r + c < rows
    lower_anti_diag = r + c >= rows - 1
    # Top half mask
    top_half = r < rows // 2

    # Left half mask
    left_half = c < cols // 2
    masks = [upper_diag, lower_diag, upper_anti_diag, lower_anti_diag,top_half, left_half]
    masks_inds = [
        ]
    for j in range(mask_data.shape[0]):
        for i in [5]:
            area = np.sum(mask_data[j, :, :] == i)
            # print(area)
            if area > 0:
                if image_data is not None:
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(image_data[j] / 255)
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(mask_data[j])
                    # plt.show()
                    if j == 7:
                        msk = upper_anti_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = im_new[np.newaxis,:]
                        new_mask = msk_new[np.newaxis,:]
                    if j == 51:
                        msk = left_half
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)

                    if j == 100:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)

                    if j == 101:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)


                    if j == 102:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)


                    if j == 127:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)
    image_data = np.concatenate((image_data,new_data),axis=0)
    mask_data = np.concatenate((mask_data,new_mask),axis=0)
    return image_data,mask_data

def Mine_resize(image = None, mask = None, final_size = None):
    """Resize image and mask to the final size."""
    image_resized = F.interpolate(image, size=(final_size[0], final_size[1]), mode="bilinear",
                                  align_corners=False)
    mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=final_size, mode="nearest").squeeze(1).long()
    return image_resized, mask_resized






class RandomOneOf(torch.nn.Module):
    def __init__(self, augmentations, p=1.0):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x  # No augmentation applied
        # Randomly select one augmentation
        idx = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[idx](x)


class KorniaAugmentation:
    def __init__(self, mode="train", num_classes=6, seed=None, size = None):
        self.mode = mode
        self.size = size
        self.num_classes = num_classes
        self.seed = seed
        torch.manual_seed(seed) if seed else None

        # Define PiecewiseAffine augmentation
        self.piecewise_affine = K.RandomThinPlateSpline(scale=0.2, align_corners=False)

        # Define Shape Transformations
        # self.affine = T.Compose([
        #     T.RandomAffine(
        #         degrees=(-179, 179),  # Rotation between -179 and 179 degrees
        #         translate=(0.01, 0.01),  # Translation up to 1% of image dimensions
        #         scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
        #         shear=(-5, 5),  # Shear range between -5 and 5 degrees
        #         # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
        #         fill=0  # Fill padding areas with black (0)
        #     )
        # ])

        # Create a RandomOneOf augmentation for the shape transformations
        self.shape_augs = torch.nn.Sequential(#RandomOneOf([
            # K.RandomThinPlateSpline(p=0.5, scale=0.1, same_on_batch=True, keepdim=True),  # Correct parameters
            T.RandomAffine(
                degrees=(-179, 179),  # Rotation between -179 and 179 degrees
                translate=(0.5, 0.5),  # Translation up to 1% of image dimensions
                scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
                shear=(-5, 5),  # Shear range between -5 and 5 degrees
                # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
                fill=1  # Fill padding areas with black (0)
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            # K.CenterCrop(size=size)
        )

        # Define Input Transformations
        self.input_augs = torch.nn.Sequential(
            RandomOneOf([
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                K.RandomMedianBlur((3, 3), p=1.0),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
            ], p=1.0),
            K.RandomBrightness(brightness=(0.9, 1.1), p=1.0),
            K.RandomContrast(contrast=(0.75, 1.25), p=1.0),
            K.RandomHue(hue=(-0.05, 0.05), p=1.0),
            K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),

        )

    def __apply_piecewise_affine(self, images, masks):
        """Apply piecewise affine transformation to both images and masks."""
        # Stack masks into additional channels

        B, C, H, W = images.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Create a temporary ones tensor to track padding
        ones_tensor = torch.ones_like(images)

        # Combine images, masks, and the ones tensor for padding tracking
        combined = torch.cat([images, masks_one_hot, ones_tensor], dim=1)

        # Apply affine transformation to the combined tensor (image + mask + ones)
        combined_augmented = self.piecewise_affine(combined)

        # Separate images, masks, and ones tensor after transformation
        images_aug = combined_augmented[:, :C]
        masks_aug = combined_augmented[:, C:C + self.num_classes]
        ones_aug = combined_augmented[:, C + self.num_classes:]


        masks_aug = masks_aug.argmax(dim=1).long()  # Convert back to class labels

        # Track padding by checking where ones_aug has been turned to 0 (padding areas)
        padding_mask = ones_aug == 0  # This will be True in the padding areas

        # Fill padding for images with 1 (where padding_mask is True)
        images_aug = torch.where(padding_mask, torch.tensor(1.0, device=images_aug.device), images_aug)
        return images_aug, masks_aug

    def __apply_erode_margins(self, masks):
        """Add margins to masks by applying erosion."""
        B, H, W = masks.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply erosion to each class
        kernel = torch.ones((3, 3), device=masks.device)
        eroded_channels = []
        for i in range(self.num_classes):
            mask_channel = masks_one_hot[:, i:i + 1]
            eroded = mask_channel
            for _ in range(5):  # Perform erosion 3 times
                eroded = KM.erosion(eroded, kernel)
            eroded_channels.append(eroded)

        # Combine eroded masks
        eroded_masks = torch.cat(eroded_channels, dim=1)
        combined_masks = eroded_masks.argmax(dim=1).long()

        return combined_masks


    def __call__(self, image, mask):
        # Step 1: Apply margins to masks
        # mask = self.__apply_erode_margins(mask)
        # Step 4: Apply input augmentations
        image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])

        # # Step 2: Apply piecewise affine transformation
        # image, mask = self.__apply_piecewise_affine(image, mask)

        # Step 3: Apply shape augmentations using RandomOneOf
        # Add the temporary tensor of ones to track padding
        ones_tensor = torch.ones_like(image)

        combined = torch.cat([image, mask.unsqueeze(1).float(), ones_tensor], dim=1)

        # Apply the augmentations to the combined tensor (image + mask + ones)
        combined_augmented = self.shape_augs(combined)

        # After augmentation, split the tensor back into the image, the mask, and the ones tensor
        image = combined_augmented[:, :image.size(1)]  # Only the image part
        mask = combined_augmented[:, image.size(1):image.size(1) + mask.unsqueeze(1).size(1)]  # Only the mask part
        ones_aug = combined_augmented[:, image.size(1) + mask.size(1):]  # Only the ones tensor

        # Set padding areas of the image to 1 (based on ones_aug tracking)
        padding_mask = ones_aug == 0  # This will be True in padding areas
        image = torch.where(padding_mask, torch.tensor(1.0, device=image.device), image)
        #
        # # Step 4: Apply input augmentations
        # image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])
        #
        # # Step 5: Combine augmented mask with the image
        # # image1 = (1 - mask) + image
        #
        # # Step 6: Resize to final size
        mask = mask.squeeze(1)
        return image, mask


def Data_class_analyze(masks,# a numpy array (batch, H, W, C)
                       class_labels): # A list of classes with correct order. for example class0 should be placed in first place.
    """This function analyzes the images and masks. it counts number and area of samples for each class.
    It is a good representation for imbalance data. It also gives hints to data augmentation and over-sampling"""

    shape = np.shape(masks)
    num_classes = len(class_labels)
    class_samples = np.zeros(num_classes)
    class_areas = np.zeros(num_classes)
    class_distribution = np.zeros((shape[0],num_classes))
    for i in range(shape[0]):
        msk = masks[i]
        if len(msk.shape)>2:
            msk = np.sum(msk, axis=2)# convert to one image
        for j in range(num_classes):
            area = np.sum(msk == j)
            if area > 0:
                # if j == 5:
                #     plt.imshow(msk)
                #     plt.show()
                class_samples[j] += 1
                class_areas[j] += area
                class_distribution[i,j] += 1
    return class_samples, class_areas, class_distribution



def split_train_val(class_samples = 0, class_distribution = 0, val_percent = 0.2):
    a = 0
    class_distribution1 = np.copy(class_distribution)
    shape = np.shape(class_distribution)
    val_samples = int(shape[0]*val_percent)
    train_samples = shape[0] - val_samples
    val_index = np.zeros(val_samples)
    train_index = np.zeros(train_samples)
    val_indexes = 0
    val_samples1 = np.copy(val_samples)
    for i in range(1,shape[1]):
        ind = np.where(class_samples[1:] == np.min(class_samples[1:]))[0]+1
        ind = ind[0]
        val = class_samples[ind]
        class_samples[ind] = np.inf
        random.seed(42)
        val_sample_inds = random.sample(range(int(val)), int(np.ceil(val*val_percent)))
        print(val_sample_inds)
        k = 0
        for m in range(shape[0]):
            if class_distribution[m,int(ind)] == 1:
                if len(np.where(np.array(val_sample_inds) == k)[0]):
                    val_index[val_indexes] = m
                    class_distribution[m] = 0*class_distribution[m]
                    val_indexes += 1
                    val_samples1 -= 1
                    if val_samples1 == 0:
                        break
                    # print(k)
                k += 1
        if val_samples1 ==0:
            break
    val_index = np.where(np.sum(class_distribution,axis = 1) == 0)
    train_index = np.where(np.sum(class_distribution, axis=1) != 0)

    # print(val_index)





    return train_index, val_index


def addsamples(images,mask,sample_th = 0.2, tissue_labels = None):
    a = 2
    angles = [45,90,135,180,225,270,325]
    shape = np.shape(images)
    shape_m = np.shape(mask)
    new_ims = []
    new_masks = []
    class_samples, class_areas, class_distribution = Data_class_analyze(mask, tissue_labels)
    average_class = class_samples / np.shape(mask)[0]
    average_area = class_areas / np.sum(class_areas)
    avg = 100 * average_class * 100 * average_area / np.sum(100 * average_class * 100 * average_area)
    new_ims = []
    new_masks = []
    for i in range(1,shape_m[3]):# start from 1 to skip background
        ind = np.where(avg[1:] == np.min(avg[1:]))[0]+1
        ind = ind[0]
        val = avg[ind]
        avg[ind] = np.inf
        class_inds = np.where(class_distribution[:,ind] == 1)
        if val < sample_th:
            up_num = int(shape[0] * sample_th/(1-sample_th) - 100*average_area[ind])
            for j in range(3):# fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break
                    new_ims.append(cv2.flip(images[inds],j-1))
                    new_masks.append(cv2.flip(mask[inds],j-1))
                    up_num -= 1

            for rot in angles:
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                    imm = cv2.warpAffine(images[inds], M, (shape[1], shape[2]),borderValue=(255,255,255))
                    new_ims.append(imm)
                    imm = cv2.warpAffine(np.array(mask[inds], dtype=np.uint8), M, (shape[1], shape[2]))
                    new_masks.append(imm)


                    up_num -= 1

            for j in range(3):  # fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    image = cv2.flip(images[inds], j-1)
                    msk = cv2.flip(mask[inds], j-1)
                    for rot in angles:
                        if up_num <= 0:
                            break
                        M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                        imm = cv2.warpAffine(image, M, (shape[1], shape[2]),borderValue=(255,255,255))
                        new_ims.append(imm)
                        imm = cv2.warpAffine(np.array(msk, dtype=np.uint8), M, (shape[1], shape[2]))
                        new_masks.append(imm)
                        up_num -= 1


            class_distribution[class_inds] = 0 * class_distribution[class_inds]
    a = 0
    images1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape[3]))
    mask1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape_m[3]))
    images1[0:shape[0]] = images
    mask1[0:shape[0]] = mask

    new_ims1 = np.array(new_ims)
    new_masks1 = np.array(new_masks, dtype=np.uint8)
    images1[shape[0]:] = new_ims1
    mask1[shape[0]:] = new_masks1
    images1 = images1.astype('float32')
    mask1 = mask1.astype('int64')

    return images1, mask1




def puma_dice_loss(preds, targets, eps=1e-5):
    num_classes = preds.shape[1]

    # Apply softmax (since preds are raw logits)
    preds = F.softmax(preds, dim=1) if num_classes > 1 else torch.sigmoid(preds)

    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Compute intersection & union
    intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
    union = torch.sum(preds + targets_one_hot, dim=(2, 3))

    # Dice score & loss
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()  # Dice loss


def compute_puma_dice_micro_dice(model = None, target_siz = None,epoch = 1, input_folder = '', output_folder = '', ground_truth_folder = '', device = None, model1=None,
                                 weights_list = None, er_di = False, augment_all = True, save_jpg = False, file_path = None,
                                 classifier_mode = False,
                                in_channels = 3
                                 ):
    # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
    # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if epoch == 1:
        if os.path.exists(output_folder):
            for root, dirs, files in os.walk(output_folder, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(output_folder)
        os.makedirs(output_folder, exist_ok=True)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define preprocessing transforms

    # Process each image
    frame_type =[]
    target_path = os.listdir(input_folder) if file_path is None else [file_path]
    for file_name in target_path:
        if file_name.endswith(".tif"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the TIF image
            image = tifffile.imread(input_path)

            # Ensure the image has 3 channels
            # Handle 4-channel images by dropping the alpha channel
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Keep only the first three channels (RGB)
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels in image: {file_name}")



            if er_di:
                disk_radius = 5
                kernel_size = (2 * disk_radius + 1, 2 * disk_radius + 1)

                # Create the circular disk structuring element
                disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                eroded_image = cv2.erode(image, disk_kernel, iterations=1)

                # Apply dilation
                dilated_image = cv2.dilate(image, disk_kernel, iterations=1)
                image = np.concatenate((image, eroded_image, dilated_image), axis=2)


            if in_channels == 4:
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/nuclei_10class_all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)

            image = cv2.resize(image, target_siz)

            image = image / 255

            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

            # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
            # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

            # Get prediction
            if weights_list != None:
                prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list)
                prediction = torch.unsqueeze(prediction, 0)
            elif augment_all:
                prediction = validate_with_augmentations(model, image_tensor)
                # prediction = prediction[0]
            else:
                prediction = model(image_tensor)
                prediction = F.softmax(prediction, dim=1)
                if model1 is not None:
                    prediction1 = model1(image_tensor)
                    prediction1 = F.softmax(prediction1, dim=1)

                    prediction = 0.5*prediction + 0.5*prediction1
            prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()

                # Post-process prediction (e.g., apply softmax or argmax)
            metas_sum = 0
            metas_class = np.empty((0,5))
            primary_sum = 0
            primary_class = np.empty((0,5))
            a_con = np.zeros((5))
            for i in range(6, 11):
                metas_sum = metas_sum + np.sum(prediction == i)
                a_con[i-6] = np.sum(prediction == i)
            metas_class = np.concatenate((metas_class,a_con[np.newaxis,:]), axis=0)

            for i in range(1, 6):
                primary_sum = primary_sum + np.sum(prediction == i)
                a_con[i-1] = np.sum(prediction == i)

            primary_class = np.concatenate((primary_class,a_con[np.newaxis,:]), axis=0)
            if primary_sum > metas_sum:
                frame_type.append([file_name ,'primary', primary_sum, metas_sum])
            else:
                frame_type.append([file_name ,'metas',primary_sum, metas_sum])
            prediction[prediction>5] = prediction[prediction>5] - 5

            colormap = {
                0: [0, 0, 0],  # Black
                1: [255, 0, 0],  # Red
                2: [0, 255, 0],  # Green
                3: [0, 0, 255],  # Blue
                4: [255, 255, 0],  # Yellow
                5: [255, 0, 255],  # Magenta
            }
            # Create an RGB image by mapping class values to colormap
            rgb_image = np.zeros((target_siz[0], target_siz[1], 3), dtype=np.uint8)
            # Save the prediction as a TIF file
            with tifffile.TiffWriter(output_path) as tif:
                tif.write(prediction.astype(np.uint8), resolution=(300, 300))

            if save_jpg:
                for class_value, color in colormap.items():
                    rgb_image[prediction == class_value] = color
                cv2.imwrite(output_path[:-4] + '.jpg', rgb_image)

            if file_path is not None:
                return prediction,primary_class,metas_class

            # print(f"Processed and saved: {file_name}")
    # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
    prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution
    with open(output_folder+'/tissue_type.txt', 'w') as file:
        for sublist in frame_type:
            file.write(" ".join(map(str, sublist)) + "\n")
    dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
                                                                         image_shape)
    np.save(output_folder+'/tissue_type_metas.npy', metas_class)
    np.save(output_folder+'/tissue_type_primary.npy', primary_class)
    # print("Overall Dice Scores:", mean_puma_dice)
    # print("Overall Mean Dice Scores:", mean_dice_classes)

    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")


    micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
                                                                         image_shape, eps=0.00001)


    return mean_puma_dice, micro_dices, mean_micro_dice


def circular_augmentation(train_images, masks, target_class, r1, r2, d1):
    """
    Apply circular augmentation to the specified class in the segmentation mask.

    Parameters:
        train_images (torch.Tensor): Input tensor of training images, size (B, C, H, W).
        masks (torch.Tensor): Input tensor of mask images, size (B, H, W) (torch.long).
        target_class (int): The class on which to apply the augmentation.
        r1 (int): Minimum radius of circles.
        r2 (int): Maximum radius of circles.
        d1 (float): Density of circles (fraction of target class area to be covered).

    Returns:
        tuple: Augmented training images and masks.
    """
    # Get device from input tensors
    device = train_images.device

    # Get dimensions
    B, C, H, W = train_images.shape
    augmented_images = train_images.clone()
    augmented_masks = masks.clone()

    for b in range(B):
        # Extract the target class region from the mask
        target_region = (masks[b] == target_class)
        target_area = target_region.sum().item()

        if target_area == 0:
            continue  # Skip if the target class is not present

        # Calculate maximum allowable circle area
        max_circle_area = target_area * d1
        current_area = 0

        circles = []  # Track placed circles as (x, y, r)

        while current_area < max_circle_area:
            # Random radius
            r = np.random.randint(r1, r2 + 1)

            # Generate a random center within the valid target region
            valid_y, valid_x = torch.where(target_region)
            if len(valid_y) == 0:
                break  # Exit if no valid points are left
            idx = np.random.randint(len(valid_y))
            x, y = valid_x[idx].item(), valid_y[idx].item()

            # Check for overlap
            overlap = False
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < r + cr:
                    overlap = True
                    break
            if not overlap:
                # Generate a circular area
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                circle_area = ((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2

                # Ensure the circle is within the target region
                circle_mask = circle_area & target_region

                new_area = circle_mask.sum().item()
                # if current_area + new_area <= max_circle_area:
                    # Apply the augmentation: set the mask to 0 and the image to 0 in the circle
                augmented_masks[b][circle_mask] = 0
                augmented_images[b][:, circle_mask] = 1

                current_area += new_area
                circles.append((x, y, r))
    return augmented_images, augmented_masks
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
def upsample_tissue(image_data = None,mask_data = None):
    rows, cols = np.shape(image_data)[1], np.shape(image_data)[2]  # Replace with your image dimensions
    r, c = np.indices((rows, cols))
    upper_diag = c >= r
    lower_diag = c <= r
    upper_anti_diag = r + c < rows
    lower_anti_diag = r + c >= rows - 1
    top_half = r < rows // 2
    left_half = c < cols // 2
    masks = [upper_diag, lower_diag, upper_anti_diag, lower_anti_diag,top_half, left_half]
    masks_inds = [0, 1, 0, 4, 2, 3, 3, 3, 5, 2, 4, 5, 4, 1, 5, 3, 4, 0, 1, 2, 4, 1, 5, 5,
                  5, 1, 3, 2, 0, 5, 4, 5, 0, 2, 4, 4, 5, 5, 4, 3, 5, 1, 4, 4, 4, 4, 5, 2,
                  2, 4, 2, 5, 4, 5, 0, 1, 4, 5, 4, 1, 5, 4, 5, 4, 2, 1, 5, 4, 5, 0, 2, 0,
                  1, 0, 4, 5, 4, 5, 4, 1, 5, 5, 5, 2, 4, 4, 4, 3, 2, 4, 5, 5, 5, 5, 0, 1,
                  3, 1, 3, 4, 5, 0, 1, 0, 4, 1, 2, 4, 2, 1, 1, 5, 4, 5, 0, 3, 1, 4, 0, 2,
                  0, 1, 0, 4, 5, 5, 2, 2, 0, 1, 4, 1, 4, 4, 0, 1, 5, 4, 4, 2, 0, 1, 5,
                  2, 5, 1, 1, 2, 5, 4, 2, 4, 4, 1, 3, 4, 2, 5, 5, 5, 5, 5, 4, 5, 2, 4, 5,
                  5, 1, 3, 1, 5, 5, 5, 5, 4, 2, 5, 0, 1, 1, 4, 5, 4, 4, 5, 1, 4, 1, 4, 0,
                  2, 0, 4, 5, 2, 5, 3, 4, 5, 5, 5, 3, 5, 4, 1 ]
    masks_inds[127] = 1
    masks_inds[100] = 1
    masks_inds[2] = 4
    masks_inds[8] = 4
    masks_inds[24] = 4
    masks_inds[73] = 5
    masks_inds[97] = 2
    masks_inds[105] = 2
    masks_inds[106] = 0
    masks_inds[108] = 1
    masks_inds[112] = 5
    masks_inds[113] = 2
    masks_inds[114] = 4
    masks_inds[115] = 0
    masks_inds[117] = 1
    masks_inds[118] = 2
    masks_inds[119] = 0
    masks_inds[123] = 1
    masks_inds[126] = 0
    masks_inds[133] = 0
    masks_inds[135] = 5
    masks_inds[136] = 4
    masks_inds[137] = 2
    masks_inds[138] = 3
    masks_inds[139] = 0
    masks_inds[192] = 2
    masks_inds[198] = 5







































    for j in range(mask_data.shape[0]):
        aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
        # if (j>114):# and ((aug_num>4) or ((np.sum(mask_data[j] == 5)>0))):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(image_data[j] / 255)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(mask_data[j])
        #     plt.show()

        msk = masks[masks_inds[j]]
        im_new = np.zeros_like(image_data[j])
        for kk in range(image_data.shape[3]):
            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
        msk_new = mask_data[j]*msk
        im_old = np.zeros_like(image_data[j])

        for kk in range(image_data.shape[3]):
            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
        msk_old = mask_data[j]*(1-msk)
        image_data[j] = im_old
        mask_data[j] = msk_old

        # if (j>114):  # and ((aug_num>4) or ((np.sum(mask_data[j] == 5)>0))):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(image_data[j] / 255)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(mask_data[j])
        #     plt.imshow(mask_data[j])
        #     plt.show()
        #     print(j)

        image_data = np.concatenate((image_data, im_new[np.newaxis, :]), axis=0)
        mask_data = np.concatenate((mask_data, msk_new[np.newaxis, :]), axis=0)


    return image_data,mask_data


def copy_data_tissue(validation_indices: List[int], data_path: str, data_path1: str, save_path: str, save_path1: str,
              data_type: str, images = None, masks = None):
    """
    Copy files from data_path to save_path based on indices and type (metastatic or primary).

    Args:
        validation_indices (List[int]): List of indices to copy.
        data_path (str): Source directory containing the files.
        save_path (str): Destination directory to save the files.
        data_type (str): Either 'metastatic' or 'primary'.
    """
    # Ensure the save_path exists
    masks[masks>5] = masks[masks>5] - 5
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    CLASS_MAPPING_TISSUE_R = {
        "tissue_stroma": 1,
        "tissue_blood_vessel": 2,
        "tissue_tumor": 3,
        "tissue_epidermis": 4,
        "tissue_necrosis": 5,
    }
    class_mapping = CLASS_MAPPING_TISSUE_R
    # Determine the prefix based on the type
    if data_type == 'primary':
        if os.path.exists(save_path):
            for root, dirs, files in os.walk(save_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path1):
            for root, dirs, files in os.walk(save_path1, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path1)
        os.makedirs(save_path1, exist_ok=True)

    prefix = f"training_set_{data_type}_roi_"

    image_shape = (1024, 1024)
    # Iterate through the indices and copy the corresponding files
    k = 0
    for index in validation_indices:
        index += 1
        file_name = f"{prefix}{index:03}_tissue.tif"  # Format index as three digits with leading zeros
        dest_file = os.path.join(save_path, file_name)

        file_name1 = f"{prefix}{index:03}.tif"  # Format index as three digits with leading zeros
        src_file1 = os.path.join(save_path1, file_name1)

        input_image = images[k]
        with tifffile.TiffWriter(src_file1) as tif:
            tif.write(np.array(input_image,dtype=np.uint8), resolution=(300, 300))

        gt_image = masks[k]
        with tifffile.TiffWriter(dest_file) as tif:
            tif.write(np.array(gt_image,dtype=np.uint8), resolution=(300, 300))

        k += 1

def validate_with_augmentations(model, image_tensor):
    """
    Perform validation with augmentations and ensemble predictions using probabilities.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.

    Returns:
        Final ensembled prediction as a NumPy array.
    """
    # Define the augmentations
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512", do_resize=False, do_rescale=False)
    def augment(tensor):
        """
        Generate 7 unique augmentations of the input tensor.
        The augmentations include:
        1. Original
        2. Rotate 90°
        3. Rotate 180°
        4. Rotate 270°
        5. Horizontal flip
        6. Horizontal flip + Rotate 90°
        7. Horizontal flip + Rotate 270°

        Args:
            tensor: Input tensor of shape (B, C, H, W).

        Returns:
            List of tensors with augmentations applied.
        """
        return [
            tensor,  # Original
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90°
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180°
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270°
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.rot90(torch.flip(tensor, dims=[3]), k=1, dims=[2, 3]),  # Horizontal flip + Rotate 90°
            torch.rot90(torch.flip(tensor, dims=[3]), k=2, dims=[2, 3]),  # Horizontal flip + Rotate 180°
            torch.rot90(torch.flip(tensor, dims=[3]), k=3, dims=[2, 3]),  # Horizontal flip + Rotate 270°
        ]

    def reverse_augment(tensor, idx):
        """
        Reverse the augmentation applied at a given index.

        Args:
            tensor: Augmented tensor of shape (C, H, W).
            idx: Index of the augmentation (0-6).

        Returns:
            Tensor with the reverse transformation applied.
        """
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Rotate 90° (reverse by rotating 270°)
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif idx == 2:  # Rotate 180° (reverse by rotating 180°)
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif idx == 3:  # Rotate 270° (reverse by rotating 90°)
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[3])
        elif idx == 5:  # Horizontal flip + Rotate 90° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3])
        elif idx == 6:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=2, dims=[2, 3]), dims=[3])
        elif idx == 7:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])
    # Apply augmentations to the input image
    augmented_inputs = augment(image_tensor)

    # Apply the model to each augmented input and store probabilities
    probabilities = []
    for augmented_input in augmented_inputs:
        with torch.no_grad():
            try:
                try:
                    model.segformer
                    device = model.device
                except:
                    model.module.segformer
                    device = model.module.device
                # Process input images
                images = processor(images=[augmented_input[0].permute(1, 2, 0).cpu().numpy()],
                                   return_tensors="pt")  # Now it's ready for SegFormer
                images = {key: value.to(device) for key, value in images.items()}
                pred = model(**images)
                pred = F.interpolate(pred.logits, size=image_tensor.size()[2:],
                                           mode='bilinear', align_corners=False)
            except:
                pred = model(augmented_input)  # Forward pass
            pred = F.softmax(pred, dim=1)  # Get probabilities
            probabilities.append(pred)

    # Reverse augmentations to align probabilities
    aligned_probabilities = [reverse_augment(prob, i) for i, prob in enumerate(probabilities)]

    # Ensemble probabilities by averaging
    stacked_probs = torch.stack(aligned_probabilities, dim=0)  # Shape: [8, C, H, W]
    averaged_probs = torch.mean(stacked_probs, dim=0)  # Shape: [C, H, W]

    # averaged_probs[0][5][averaged_probs[0][5]>0.1] = 1
    # Final prediction: Argmax over the averaged probabilities
    final_prediction = torch.argmax(averaged_probs, dim=1)  # Shape: [H, W]

    # Move to CPU and convert to NumPy
    return averaged_probs


def validate_with_augmentations1(model, image_tensor):
    """
    Perform validation with augmentations and ensemble predictions.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.

    Returns:
        Final ensembled prediction as a NumPy array.
    """
    # Define the augmentations
    def augment(tensor):
        return [
            tensor,  # Original
            torch.flip(tensor, dims=[2]),  # Vertical flip
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.flip(tensor, dims=[2, 3]),  # Vertical + Horizontal flip
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90 degrees
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180 degrees
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270 degrees
            torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])  # Rotate 90 + Horizontal flip
        ]

    # Apply augmentations to the input image
    augmented_inputs = augment(image_tensor)

    # Apply the model to each augmented input and store predictions
    predictions = []
    for augmented_input in augmented_inputs:
        pred = model(augmented_input)  # Forward pass
        pred = F.softmax(pred, dim=1)  # Apply softmax
        pred = torch.argmax(pred, dim=1)  # Get class predictions
        predictions.append(pred)

    # Reverse augmentations to align predictions
    def reverse_augment(tensor, idx):
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Vertical flip
            return torch.flip(tensor, dims=[1])
        elif idx == 2:  # Horizontal flip
            return torch.flip(tensor, dims=[2])
        elif idx == 3:  # Vertical + Horizontal flip
            return torch.flip(tensor, dims=[1, 2])
        elif idx == 4:  # Rotate 90 degrees
            return torch.rot90(tensor, k=3, dims=[1, 2])
        elif idx == 5:  # Rotate 180 degrees
            return torch.rot90(tensor, k=2, dims=[1, 2])
        elif idx == 6:  # Rotate 270 degrees
            return torch.rot90(tensor, k=1, dims=[1, 2])
        elif idx == 7:  # Rotate 90 + Horizontal flip
            return torch.rot90(torch.flip(tensor, dims=[2]), k=3, dims=[1, 2])

    # Align all augmented predictions
    aligned_predictions = [reverse_augment(pred, i) for i, pred in enumerate(predictions)]

    # Ensemble predictions by majority voting
    stacked_preds = torch.stack(aligned_predictions, dim=0)
    final_prediction = torch.mode(stacked_preds, dim=0).values  # Majority vote

    # Move to CPU and convert to NumPy
    return final_prediction.cpu().numpy()

def validate_with_augmentations_and_ensembling(model, image_tensor, weights_list=None, device = ''):
    """
    Perform validation with test-time augmentations (TTA), ensembling, and debugging visualizations.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored on GPU.
        weights_list: List of paths to model weight files for ensembling.

    Returns:
        Final ensembled prediction as a NumPy array.
    """

    # Define the augmentations
    def augment(tensor):
        """
        Generate 7 unique augmentations of the input tensor.
        The augmentations include:
        1. Original
        2. Rotate 90°
        3. Rotate 180°
        4. Rotate 270°
        5. Horizontal flip
        6. Horizontal flip + Rotate 90°
        7. Horizontal flip + Rotate 270°

        Args:
            tensor: Input tensor of shape (B, C, H, W).

        Returns:
            List of tensors with augmentations applied.
        """
        return [
            tensor,  # Original
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90°
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180°
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270°
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.rot90(torch.flip(tensor, dims=[3]), k=1, dims=[2, 3]),  # Horizontal flip + Rotate 90°
            torch.rot90(torch.flip(tensor, dims=[3]), k=2, dims=[2, 3]),  # Horizontal flip + Rotate 180°
            torch.rot90(torch.flip(tensor, dims=[3]), k=3, dims=[2, 3]),  # Horizontal flip + Rotate 270°
        ]

    def reverse_augment(tensor, idx):
        """
        Reverse the augmentation applied at a given index.

        Args:
            tensor: Augmented tensor of shape (C, H, W).
            idx: Index of the augmentation (0-6).

        Returns:
            Tensor with the reverse transformation applied.
        """
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Rotate 90° (reverse by rotating 270°)
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif idx == 2:  # Rotate 180° (reverse by rotating 180°)
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif idx == 3:  # Rotate 270° (reverse by rotating 90°)
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[3])
        elif idx == 5:  # Horizontal flip + Rotate 90° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3])
        elif idx == 6:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=2, dims=[2, 3]), dims=[3])
        elif idx == 7:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])
    # Initialize final probability predictions
    model.to(device)
    model.eval()
    final_probabilities = None  # Will hold the ensembled probabilities

    # Iterate through each model weight
    for k, weight_path in enumerate(weights_list):
        # Load model weights
        model.load_state_dict(torch.load(weight_path))

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_probabilities = []
        for i, augmented_input in enumerate(augmented_inputs):
            augmented_input = augmented_input.to(device)
            pred = model(augmented_input)  # Forward pass
            prob = F.softmax(pred, dim=1)  # Get probabilities
            augmented_probabilities.append(prob)

            # Debugging: Visualize forward-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Forward Augmentation {i}")
            # plt.imshow(prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()

        # Reverse augmentations to align predictions
        aligned_probabilities = []
        for i, prob in enumerate(augmented_probabilities):
            reversed_prob = reverse_augment(prob, i)
            aligned_probabilities.append(reversed_prob)

            # Debugging: Visualize reverse-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Reversed Augmentation {i}")
            # plt.imshow(reversed_prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()

        # Ensure alignment of dimensions
        aligned_probabilities = torch.stack([p.squeeze(0) for p in aligned_probabilities], dim=0)

        # Average probabilities across augmentations for TTA
        tta_probability = torch.mean(aligned_probabilities, dim=0)

        # Add TTA probability to the ensemble
        if final_probabilities is None:
            final_probabilities = tta_probability
        else:
            final_probabilities += tta_probability

    # Average probabilities across all models in the ensemble
    final_probabilities /= len(weights_list)
    # final_probabilities[5][final_probabilities[5]<0.6] = 0
    # Final prediction (class-wise argmax)
    final_predictions = torch.argmax(final_probabilities, dim=0)

    # Move to CPU and convert to NumPy
    return final_probabilities



def validate_with_augmentations_and_ensembling1(model, image_tensor, weights_list=None,device = ''):
    """
    Perform validation with test-time augmentations and model ensembling.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.
        weights_list: List of paths to model weight files for ensembling.

    Returns:
        Final ensembled prediction as a NumPy array.
    """

    # Define the augmentations
    def augment(tensor):
        return [
            tensor,  # Original
            torch.flip(tensor, dims=[2]),  # Vertical flip
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.flip(tensor, dims=[2, 3]),  # Vertical + Horizontal flip
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90 degrees
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180 degrees
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270 degrees
            torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])  # Rotate 90 + Horizontal flip
        ]

    # Reverse augmentations to align predictions
    def reverse_augment(tensor, idx):
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Vertical flip
            return torch.flip(tensor, dims=[1])
        elif idx == 2:  # Horizontal flip
            return torch.flip(tensor, dims=[2])
        elif idx == 3:  # Vertical + Horizontal flip
            return torch.flip(tensor, dims=[1, 2])
        elif idx == 4:  # Rotate 90 degrees
            return torch.rot90(tensor, k=3, dims=[1, 2])
        elif idx == 5:  # Rotate 180 degrees
            return torch.rot90(tensor, k=2, dims=[1, 2])
        elif idx == 6:  # Rotate 270 degrees
            return torch.rot90(tensor, k=1, dims=[1, 2])
        elif idx == 7:  # Rotate 90 + Horizontal flip
            return torch.rot90(torch.flip(tensor, dims=[2]), k=3, dims=[1, 2])

    # Initialize final probability predictions
    model.to(device)
    model.eval()
    final_probabilities = None  # Will hold the ensembled probabilities
    preds = []
    # Iterate through each model weight
    for k, weight_path in enumerate(weights_list):
        # Load model weights
        model.load_state_dict(torch.load(weight_path))
        model.eval()

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_predictions = []
        for augmented_input in augmented_inputs:
            augmented_input = augmented_input.to(device)
            pred = model(augmented_input)  # Forward pass
            pred = F.softmax(pred, dim=1)  # Apply softmax
            augmented_predictions.append(pred)

        # Reverse augmentations to align predictions
        aligned_predictions = [
            reverse_augment(torch.argmax(pred, dim=1), i) for i, pred in enumerate(augmented_predictions)
        ]

        # Aggregate augmented predictions
        stacked_preds = torch.stack(aligned_predictions, dim=0)
        tta_prediction = torch.mode(stacked_preds, dim=0).values  # Majority vote for TTA

        # Add TTA prediction to the ensemble
        preds.append(tta_prediction)

    stacked_preds = torch.stack(preds, dim=0)
    final_predictions = torch.mode(stacked_preds, dim=0).values  # Majority vote for TTA

    return final_predictions[0].cpu().numpy()

def compute_puma_dice_micro_dice_prediction(model = None, target_siz = None,input_path = '', device = None, weights_list = None, augment_all = True):
    # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
    # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    # Read the TIF image
    image = tifffile.imread(input_path)

    # Ensure the image has 3 channels
    # Handle 4-channel images by dropping the alpha channel
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Keep only the first three channels (RGB)
    elif image.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels in image:")


    # image = cv2.resize(image, target_siz)

    image = image / 255

    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
    # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

    # Get prediction
    if weights_list != None:
        prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list, device=device)
    elif augment_all:
        prediction = validate_with_augmentations(model, image_tensor)
        # prediction = prediction[0]
    else:
        prediction = model(image_tensor)
        prediction = F.softmax(prediction, dim=1)
    prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()

    # print(f"Processed and saved: {file_name}")


    return prediction

def save_prediction_for_dice(output_folder = '', file_name = '',prediction = None, save_jpg = False):
    # Post-process prediction (e.g., apply softmax or argmax)
    output_path = os.path.join(output_folder, file_name)

    colormap = {
        0: [0, 0, 0],  # Black
        1: [255, 0, 0],  # Red
        2: [0, 255, 0],  # Green
        3: [0, 0, 255],  # Blue
        4: [255, 255, 0],  # Yellow
        5: [255, 0, 255],  # Magenta
    }

    # Create an RGB image by mapping class values to colormap
    rgb_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Save the prediction as a TIF file
    with tifffile.TiffWriter(output_path) as tif:
        tif.write(prediction.astype(np.uint8), resolution=(300, 300))

    if save_jpg:
        for class_value, color in colormap.items():
            rgb_image[prediction == class_value] = color
        cv2.imwrite(output_path[:-4] + '.jpg', rgb_image)


def compute_puma_dice_micro_dice_from_folder(output_folder = '', ground_truth_folder = ''):
    # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
    prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution

    dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
                                                                         image_shape)
    # print("Overall Dice Scores:", mean_puma_dice)
    # print("Overall Mean Dice Scores:", mean_dice_classes)

    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")


    micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
                                                                         image_shape, eps=0.00001)
    return mean_puma_dice, micro_dices, mean_micro_dice

def fill_background_holes_batch(masks, max_hole_size=5000):
    """
    Fill background holes for each class in a batch of labeled masks, with a size threshold.

    Args:
        masks (numpy.ndarray): Input labeled masks of shape (N, H, W) with values [0-5].
        max_hole_size (int): Maximum size of holes to be filled.

    Returns:
        numpy.ndarray: Masks with background holes filled, same shape as input.
    """
    # Create an output array with the same shape as input
    filled_masks = masks.copy()

    # Loop through each mask in the batch
    for i in range(masks.shape[0]):
        mask = masks[i]  # Current mask (H, W)

        # Get unique class labels (excluding background, 0)
        class_labels = np.unique(mask)
        class_labels = class_labels[class_labels != 0]

        for cls in class_labels:
            # Create a binary mask for the current class
            class_binary = (mask == cls).astype(np.uint8) * 255

            # Invert the class binary to identify background holes
            inverted_mask = cv2.bitwise_not(class_binary)

            # Find connected components in the inverted mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)

            # Iterate through connected components (excluding the background, label 0)
            for label in range(1, num_labels):
                # Get the size of the current hole
                hole_size = stats[label, cv2.CC_STAT_AREA]

                # Fill the hole only if its size is less than or equal to the threshold
                if hole_size <= max_hole_size:
                    class_binary[labels == label] = 255

            # Update the original mask with the filled regions for the current class
            filled_masks[i][class_binary > 0] = cls

    return filled_masks


def modify_model_for_new_classes(model = None, n_classes = None):
    decoder_output_channels = model.segmentation_head[0].in_channels
    model.segmentation_head = smp.base.SegmentationHead(
        in_channels=decoder_output_channels,  # Decoder's final output channels
        out_channels=n_classes,  # Number of new classes
        activation=None  # Raw logits
    )
    # Initialize new segmentation head weights
    def initialize_head_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    model.segmentation_head.apply(initialize_head_weights)
    return model


def random_sub_image_sampling(image_batch: torch.Tensor, mask_batch: torch.Tensor, sub_size: int):
    """
    Randomly selects sub-images from the image and mask batches.

    Args:
        image_batch (torch.Tensor): Batch of images with shape (B, C, H, W)
        mask_batch (torch.Tensor): Batch of masks with shape (B, H, W)
        sub_size (int): Size of the sub-image (assumes square sub-images)

    Returns:
        torch.Tensor: Sub-image batch of shape (B, C, sub_size, sub_size)
        torch.Tensor: Sub-mask batch of shape (B, sub_size, sub_size)
    """
    B, C, H, W = image_batch.shape
    _, H_m, W_m = mask_batch.shape

    assert H == H_m and W == W_m, "Image and mask dimensions must match."
    assert sub_size <= H and sub_size <= W, "Sub-image size must be within the image dimensions."

    # Randomly select top-left coordinates for cropping for each image independently
    top = torch.randint(0, H - sub_size + 1, (B,))
    left = torch.randint(0, W - sub_size + 1, (B,))

    # Extract sub-images and sub-masks
    sub_images = torch.stack(
        [image_batch[i, :, top[i]:top[i] + sub_size, left[i]:left[i] + sub_size] for i in range(B)])
    sub_masks = torch.stack([mask_batch[i, top[i]:top[i] + sub_size, left[i]:left[i] + sub_size] for i in range(B)])

    return sub_images, sub_masks

def add_classes_metas(mask_data_metas = None, num_classes = 6):
    for i in range(num_classes):
        mask_data_metas[mask_data_metas == 1] = 6
        mask_data_metas[mask_data_metas == 2] = 7
        mask_data_metas[mask_data_metas == 3] = 8
        mask_data_metas[mask_data_metas == 5] = 10

    return mask_data_metas


def adapt_checkpoint(checkpoint, model):
    model_dict = model.state_dict()
    new_checkpoint = {}
    for key, value in checkpoint.items():
        # print(f"Skipping {key} due to shape mismatch")
        if key in model_dict and model_dict[key].shape == value.shape:
            new_checkpoint[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch")
    return new_checkpoint

def copy_data(validation_indices: List[int], data_path: str, data_path1: str, save_path: str, save_path1: str, data_type: str):
    """
    Copy files from data_path to save_path based on indices and type (metastatic or primary).

    Args:
        validation_indices (List[int]): List of indices to copy.
        data_path (str): Source directory containing the files.
        save_path (str): Destination directory to save the files.
        data_type (str): Either 'metastatic' or 'primary'.
    """
    # Ensure the save_path exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    CLASS_MAPPING_TISSUE_R = {
        "tissue_stroma": 1,
        "tissue_blood_vessel": 2,
        "tissue_tumor": 3,
        "tissue_epidermis": 4,
        "tissue_necrosis": 5,
    }
    class_mapping = CLASS_MAPPING_TISSUE_R
    # Determine the prefix based on the type
    if data_type == 'primary':
        if os.path.exists(save_path):
            for root, dirs, files in os.walk(save_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path1):
            for root, dirs, files in os.walk(save_path1, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path1)
        os.makedirs(save_path1, exist_ok=True)



    prefix = f"training_set_{data_type}_roi_"

    image_shape = (1024,1024)
    # Iterate through the indices and copy the corresponding files
    for index in validation_indices:
        index+=1
        file_name = f"{prefix}{index:03}_tissue.geojson"  # Format index as three digits with leading zeros
        src_file = os.path.join(data_path, file_name)
        dest_file = os.path.join(save_path, file_name)

        file_name1 = f"{prefix}{index:03}.tif"  # Format index as three digits with leading zeros
        src_file1 = os.path.join(data_path1, file_name1)


        if os.path.exists(src_file):
            file_path = src_file
            with open(file_path, 'r') as geojson_file:
                try:
                    data = json.load(geojson_file)
                except json.JSONDecodeError:
                    print(f"Skipping invalid GeoJSON file: {file_name}")
                    continue

                # Create temporary maps for the current GeoJSON file
                current_class_map = np.zeros(image_shape, dtype=np.uint8)
                current_instance_map = np.zeros(image_shape, dtype=np.uint32)
                # Iterate over features in the GeoJSON
                for i, feature in enumerate(data['features']):
                    geometry = shape(feature['geometry'])
                    class_name = feature['properties']['classification']['name']

                    # Rasterize the geometry onto the instance and class maps
                    mask = rasterize(
                        [(geometry, 1)],
                        out_shape=image_shape,
                        fill=0,
                        default_value=1,
                        dtype=np.uint8
                    )
                    current_instance_map[mask == 1] = i + 1  # Assign unique instance IDs
                    if class_name in class_mapping:
                        current_class_map[mask == 1] = class_mapping[class_name]

                tif_file_name = f"{prefix}{index:03}_tissue.tif"
                tif_save_path = os.path.join(save_path, tif_file_name)

                min_val, max_val = current_class_map.min(), current_class_map.max()

                with tifffile.TiffWriter(tif_save_path) as tif:
                    tif.write(
                        current_class_map,
                        resolution=(300, 300),  # Set resolution to 300 DPI for both X and Y
                        extratags=[
                            ('MinSampleValue', 'I', 1, int(min_val)),
                            ('MaxSampleValue', 'I', 1, int(max_val)),
                        ]
                    )

                # print(f"Saved TIFF: {tif_save_path}")
            shutil.copy(src_file1, save_path1)
            # print(f"Copied: {src_file} -> {save_path1}")
        else:
            a= 0
            # print(f"File not found: {src_file}")


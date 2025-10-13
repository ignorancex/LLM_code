"""
Electrolyzers-HSI Dataset Loading Utilities

This module provides a comprehensive set of utility functions for handling, processing,
visualizing, and evaluating hyperspectral image (HSI) data.

Key Functional Areas:

1. Spectral Processing:
   - `spectral_binning`: Reduces spectral resolution by averaging adjacent bands.

2. Data Loading and Preprocessing:
   - `Generate_data`: Loads selected hyperspectral cubes, segmentation masks, and RGB images.
   - `load_triplets`: Reads HSI-RGB-GT triplets from disk structured in numbered subfolders.

3. Visualization:
   - `plot_hsi_rgb_mask_triples`: Visualizes false-color HSI, RGB images, and ground truth masks.
   - `visualize`: Displays a single segmentation mask with a custom colormap.

4. Feature Extraction and Augmentation:
   - `extract_features_and_labels`: Converts HSI data and masks into labeled vector pairs.
   - `augment_data`: Applies a series of augmentations to spectral feature vectors.

5. Evaluation:
   - `evaluate_segmentation`: Computes confusion matrix, accuracy, precision, recall, F1 score,
     IoU, Dice, and kappa for semantic segmentation results.
   - `evaluate_segmentation2`: Similar to `evaluate_segmentation`, returns metrics in dictionary form.

6. Normalization:
   - `normalize_hsi_bandwise_with_mask`: Normalizes each spectral band independently, only over masked pixels.
   - `normalize_hsi_vector_wise_with_mask`: Normalizes each spectral vector (per pixel) independently
     over masked pixels.

Dependencies:
- NumPy, OpenCV, Matplotlib, Spectral Python (SPy), SciKit-Learn

This file is designed to support HSI segmentation workflows. Each function is compatible with
NumPy arrays and integrates well with PyTorch-based machine learning pipelines.

Example:
    cubes, masks, rgb = Generate_data(indices, hsi_list, mask_list, rgb_list)
    plot_hsi_rgb_mask_triples(cubes, rgb, masks)

Author: HiF-Explo: Elias Arbash
"""
import os 
import spectral
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import *

def spectral_binning(hsi_list):
    """
    Perform spectral binning by averaging every two adjacent bands in each hyperspectral image.

    Parameters:
    hsi_list (list of np.array): List of 3D hyperspectral images (height × width × bands)

    Returns:
    list of np.array: List of binned hyperspectral images with reduced spectral resolution
    """
    binned_hsi_list = []
    
    for hsi in hsi_list:
        height, width, bands = hsi.shape

        # Ensure the number of bands is even for proper binning
        if bands % 2 != 0:
            hsi = hsi[:, :, :-1]  # Drop the last band if odd

        # Perform spectral binning (average every two adjacent bands)
        binned_hsi = (hsi[:, :, 0::2] + hsi[:, :, 1::2]) / 2
        
        binned_hsi_list.append(binned_hsi)

    return binned_hsi_list


def Generate_data(data_list, HSI_cubes, seg_masks,RGB_img):
    """
    Reading PCB-Vision validation and testing HS cubes.
    It does not perform any augmentation
    
    Args:
        data_list (list): A list of indices corresponding to the HS cubes 
                          in PCB-Vision to be read.
        HSI_cubes (list): A list of the augmented HS cubes 
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): HS cubes
        masks (list): segmentation masks
    
    """
    cubes = []
    masks = []
    RGB = []
    for i, ii in enumerate(data_list):
        cubes.append(HSI_cubes[ii])
        masks.append(seg_masks[ii])
        RGB.append(RGB_img[ii])
        
    return cubes, masks, RGB



def plot_hsi_rgb_mask_triples(hsi_list, rgb_list, masks_list):
    """
    Plot triples of HSI (false-color), RGB, and ground truth mask side by side.

    Parameters:
        hsi_list: List of hyperspectral images (NumPy arrays of shape (H, W, B))
        rgb_list: List of RGB images (NumPy arrays of shape (H, W, 3))
        masks_list: List of ground truth masks (NumPy arrays of shape (H, W))
    """
    # Determine the number of triples to plot
    num_samples = len(hsi_list)
    
    # Create a figure with 3 columns per triple (HSI, RGB, mask)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Ensure axes is a 2D array for consistency (even if there's only one sample)
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Define class names and colors for the mask colormap
    class_names = ['Background', 'Mesh', 'Steel_Black', 'Steel_Grey', 'HTEL_Grey', 'HTEL_Black']
    colors = ['#000000', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # Example colors
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0, 7), cmap.N)
    # Plot each triple
    for i in range(num_samples):
        hsi = hsi_list[i]
        rgb = rgb_list[i]
        mask = masks_list[i]
        
        # Create a false-color image using selected bands.
        # Note: The code below uses bands 320 (R), 270 (G), and 200 (B); adjust these indices as needed.
        false_color_img = hsi[:, :, (320, 270, 200)]
        
        # Normalize the false-color image to the range [0, 1] for visualization
        false_color_img = (false_color_img - np.min(false_color_img)) / (
            np.max(false_color_img) - np.min(false_color_img) + 1e-6)
        
        # Plot the false-color HSI image
        axes[i, 0].imshow(false_color_img)
        axes[i, 0].set_title(f"HSI False-Color {i+1}")
        axes[i, 0].axis('on')

        # Plot the RGB image
        axes[i, 1].imshow(rgb)
        axes[i, 1].set_title(f"RGB Image {i+1}")
        axes[i, 1].axis('on')

        # Plot the ground truth mask with a defined colormap and normalization
        cax = axes[i, 2].imshow(mask, cmap=cmap, norm=norm, interpolation='none')
        axes[i, 2].set_title(f"Ground Truth Mask {i+1}")
        axes[i, 2].axis('on')

        # Create a colorbar for the mask plot using the figure's colorbar method
        cbar = fig.colorbar(cax, ax=axes[i, 2], ticks=np.arange(0.5, 6.5, 1))
        cbar.ax.set_yticklabels(class_names, fontsize=10)
    
    # Adjust the layout and display the complete figure once
    plt.tight_layout()
    plt.show()
    

def load_triplets(output_dir):
    """
    Loads triplets of images from numbered subfolders in the output_dir.
    
    Each subfolder is expected to contain:
      - "HSI.hdr" (with the accompanying ENVI binary file), saved using spectral.envi.save_image.
      - "RGB.jpg", an RGB image saved as JPEG.
      - "GT.jpg", a ground truth mask saved as JPEG.
      
    The function reads these files, converts them to appropriate formats,
    and returns three lists:
        HSI_images: list of HSI images (numpy arrays).
        RGB_images: list of RGB images (numpy arrays with channel order R, G, B).
        GT_images:  list of GT mask images (numpy arrays).
    
    Parameters:
        output_dir (str): Base directory containing numbered subfolders.
        
    Returns:
        tuple: (HSI_images, RGB_images, GT_images)
    """
    HSI_images = []
    RGB_images = []
    GT_images = []
    
    # List and sort subfolders (assuming folder names are integers as strings).
    subfolders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    subfolders = sorted(subfolders, key=lambda x: int(x))
    
    for folder in subfolders:
        folder_path = os.path.join(output_dir, folder)
        
        # Load the HSI image using spectral library.
        hsi_hdr_path = os.path.join(folder_path, 'HSI.hdr')
        # Open the ENVI file. The returned object supports the .load() method to read the full array.
        hsi_obj = spectral.envi.open(hsi_hdr_path)
        hsi_img = hsi_obj.load()  # Reads the entire HSI image into memory.
        HSI_images.append(hsi_img[:,:,50:-40])
        
        # Load the RGB image using OpenCV.
        rgb_path = os.path.join(folder_path, 'RGB.jpg')
        rgb_img = cv2.imread(rgb_path)
        # Convert from BGR (OpenCV default) to RGB.
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        RGB_images.append(rgb_img)
        
        # Load the GT mask image as grayscale.
        gt_path = os.path.join(folder_path, 'GT.png')
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        GT_images.append(gt_img)
    
    return HSI_images, RGB_images, GT_images

def visualize(mask):
    """Mask visualization

    Show a PCB-vision mask in using specific colormap.
    
    Parameters:
        mask (numpy.ndarray): the 2D mask image.
    """

    class_names = ['Background', 'Mesh', 'Steel_Black', 'Steel_Grey', 'HTEL_Grey', 'HTEL_Black']

    # Define the colors for each class
    colors = ['#000000', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # Example colors

    # Create a colormap
    cmap = ListedColormap(colors)

    # Create a normalization object
    norm = BoundaryNorm(np.arange(0, 7), cmap.N)

    # Plot the ground truth map
    fig, ax = plt.subplots(figsize = [16,12])
    cax = ax.imshow(mask, cmap=cmap, norm=norm, interpolation='none')# test_GT
    # Hide the axes
    #ax.axis('off')
    # Create the color bar
    cbar = fig.colorbar(cax, ticks=np.arange(0.5, 6.5, 1))
    cbar.ax.set_yticklabels(class_names, fontsize=16)
    
############################################################

# This section is for ML files functions
def extract_features_and_labels(hsi_input, mask_input, ignore_classes=[0]):
    """
    Extract feature vectors and labels from HSI image and mask, ignoring specified classes.
    
    Parameters:
        hsi (numpy.ndarray): The HSI image.
        mask (numpy.ndarray): The corresponding ground truth mask.
        ignore_classes (list, optional): List of classes to be ignored. Defaults to [0].
    
    Returns:
        X (numpy.ndarray): Array of feature vectors.
        y (numpy.ndarray): Array of labels.
    """
    X = []
    y = []
    
    if isinstance(hsi_input,list) and isinstance(mask_input,list):
        hsi_list = hsi_input
        mask_list = mask_input
    else:
        hsi_list = [hsi_input]
        mask_list = [mask_input]
        
    for hsi,mask in zip(hsi_list, mask_list):
        h, w, c = hsi.shape
        
        for i in range(h):
            for j in range(w):
                label = mask[i, j]
                if label not in ignore_classes:
                    X.append(hsi[i, j, :])
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y   

def augment_data(vector, label, augmentations, times):
    augmented_X = []
    augmented_y = []
    for _ in range(times):
        for x,y in zip(vector,label):
            augmented_X.append(x)
            augmented_y.append(y)
            x_aug = x.copy()
            for aug in augmentations:
                x_aug = aug(x_aug)
            augmented_X.append(x_aug)
            augmented_y.append(y)
            
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    return augmented_X, augmented_y


###########################################################################


from sklearn.metrics import confusion_matrix
def evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes): # Yes Please
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    # Calculate pixel accuracy per class
    pixel_accuracy_per_class = true_positive_sum / (true_positive_sum + false_negative_sum)

    # Calculate pixel accuracy
    pixel_accuracy = np.sum(true_positive_sum) / np.sum(confusion_matrix_sum)

    # Calculate precision, recall, F1 score
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)
    recall = true_positive_sum / (true_positive_sum + false_negative_sum)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Calculate Intersection Over Union (IoU)
    iou = intersection_sum / union_sum

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_sum) / (np.sum(confusion_matrix_sum, axis=1) + np.sum(confusion_matrix_sum, axis=0))

    # Calculate Kappa coefficient
    total_pixels = np.sum(confusion_matrix_sum)
    observed_accuracy = np.sum(true_positive_sum) / total_pixels
    expected_accuracy = np.sum(true_positive_sum) * np.sum(confusion_matrix_sum, axis=1) / total_pixels**2
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    # Return the calculated evaluation metrics
    return confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class,   pixel_accuracy, iou, dice_coefficient, kappa


def evaluate_segmentation2(ground_truth_masks, predicted_masks, num_classes):
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    # Calculate pixel accuracy per class
    pixel_accuracy_per_class = true_positive_sum / (true_positive_sum + false_negative_sum)

    # Calculate pixel accuracy (Overall Accuracy)
    pixel_accuracy = np.sum(true_positive_sum) / np.sum(confusion_matrix_sum)

    # Calculate precision, recall, F1 score
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)
    recall = true_positive_sum / (true_positive_sum + false_negative_sum)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Calculate Intersection Over Union (IoU)
    iou = intersection_sum / union_sum

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_sum) / (np.sum(confusion_matrix_sum, axis=1) + np.sum(confusion_matrix_sum, axis=0))

    # Calculate Kappa coefficient for all classes together (GitHub Repo Method)
    total_pixels = np.sum(confusion_matrix_sum)
    number = np.trace(confusion_matrix_sum)
    sum_product = np.sum(np.sum(confusion_matrix_sum, axis=0) * np.sum(confusion_matrix_sum, axis=1))
    pe = sum_product / (total_pixels ** 2)
    kappa = (pixel_accuracy - pe) / (1 - pe)

    # Calculate Average Accuracy (AA)
    average_accuracy = np.mean(pixel_accuracy_per_class)

    # Return the calculated evaluation metrics
    return {
        "confusion_matrix_sum": confusion_matrix_sum,
        "true_positive_sum": true_positive_sum,
        "true_negative_sum": true_negative_sum,
        "false_positive_sum": false_positive_sum,
        "false_negative_sum": false_negative_sum,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "pixel_accuracy_per_class": pixel_accuracy_per_class,
        "pixel_accuracy (OA)": pixel_accuracy,  # Overall Accuracy (OA)
        "iou": iou,
        "dice_coefficient": dice_coefficient,
        "kappa": kappa,  # Overall Kappa for all classes
        "average_accuracy (AA)": average_accuracy  # Average Accuracy (AA)
    }

def normalize_hsi_bandwise_with_mask(hsi, mask):
    # Ensure that hsi and mask are NumPy arrays
    hsi = np.asarray(hsi)
    mask = np.asarray(mask)
    
    normalized_hsi = np.copy(hsi)
    for b in range(hsi.shape[2]):  # iterate over bands
        # select only relevant pixels where mask > 0
        relevant_pixels = hsi[:, :, b][mask > 0]
        # If there are no relevant pixels, skip this band
        if relevant_pixels.size == 0:
            continue
        min_val = np.min(relevant_pixels)
        max_val = np.max(relevant_pixels)
        # Avoid division by zero in case max_val equals min_val
        if max_val - min_val > 0:
            normalized_hsi[:, :, b] = (hsi[:, :, b] - min_val) / (max_val - min_val)
        else:
            normalized_hsi[:, :, b] = 0
        # Ensure the background remains zero after normalization
        normalized_hsi[:, :, b][mask == 0] = 0
    return normalized_hsi

def normalize_hsi_vector_wise_with_mask(hsi, mask):
    # Ensure that hsi and mask are NumPy arrays
    hsi = np.asarray(hsi)
    mask = np.asarray(mask)
    
    normalized_hsi = np.copy(hsi)
    
    # Reshape hsi to have shape (height * width, bands) for vector-wise processing
    h, w, b = hsi.shape
    hsi_reshaped = hsi.reshape(-1, b)  # Flatten spatial dimensions into rows, keeping bands as columns
    # Apply mask to get relevant pixels
    masked_indices = np.where(mask.reshape(-1) > 0)[0]
    if masked_indices.size == 0:
        return normalized_hsi  # Return original hsi if no relevant pixels
    
    relevant_pixels = hsi_reshaped[masked_indices]  # Get all relevant pixel vectors
    
    # For each pixel vector, compute min and max across all bands
    for i in range(relevant_pixels.shape[0]):
        pixel_vector = relevant_pixels[i]
        min_val = np.min(pixel_vector)
        max_val = np.max(pixel_vector)
        # Avoid division by zero in case max_val equals min_val
        if max_val - min_val > 0:
            normalized_hsi_reshaped = (relevant_pixels[i] - min_val) / (max_val - min_val)
            hsi_reshaped[masked_indices[i]] = normalized_hsi_reshaped
        else:
            hsi_reshaped[masked_indices[i]] = 0
    
    # Reshape back to original shape
    normalized_hsi = hsi_reshaped.reshape(h, w, b)
    
    # Ensure the background remains zero after normalization
    for b in range(normalized_hsi.shape[2]):
        normalized_hsi[:, :, b][mask == 0] = 0
    
    return normalized_hsi
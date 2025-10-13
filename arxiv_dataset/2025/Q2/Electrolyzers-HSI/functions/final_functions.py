"""
Electrolyzers-HSI Inference Pipeline Utilities

This module provides the set of tools for the inference pipeline of hyperspectral image (HSI) segmentation workflows.
It includes functionality for preprocessing, feature extraction, segmentation refinement, majority voting,
visualization, and performance evaluation.

Main Functional Areas:

1. **Segmentation Refinement and Voting**:
   - `majority_voting_with_segmentation`: Refines predicted masks using Masking-style segmentations (https://github.com/hifexplo/Masking) 
     and majority voting.
   - `apply_majority_voting`: Applies majority class voting to connected components in a predicted mask.

2. **Serialization and Mask Handling**:
   - `load_and_decompress`: Loads and decompresses packed segmentation masks stored using `pickle` and `gzip`.
   - `reconstruct_predictions`: Reconstructs full-size predicted masks from flattened predictions and masks.

3. **Metrics and Evaluation**:
   - `calculate_segmentation_metrics`: Computes metrics like F1-score, Average Accuracy (AA), Overall Accuracy (OA), and Cohen’s Kappa.
   
4. **Neighborhood Feature Expansion**:
   - `gain_neighborhood_band`: Expands spectral features to include GSE (Group Spectral Embedding) context for SpectralFormer input.

5. **Feature Extraction**:
   - `extract_features_and_labels`: Extracts spectral vectors and corresponding labels from HSI and ground truth masks.

6. **Visualization**:
   - `visualize_mask`: Displays a color-coded segmentation mask using a predefined class colormap.

Each function is designed to work with NumPy arrays and can be integrated into both traditional machine learning and deep learning pipelines.

Typical Use Cases:
- Post-processing semantic segmentation predictions.
- Evaluating segmentation models using standard classification metrics.
- Preparing HSI features and labels for training or testing.
- Visualizing segmentation outputs for qualitative assessment.

Dependencies:
- NumPy
- SciPy
- scikit-image
- scikit-learn
- Matplotlib

Author: HiF-Explo: Elias Arbash
"""
import numpy as np
from skimage.measure import label
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

import pickle
import gzip
from scipy.stats import mode

def majority_voting_with_segmentation(prediction_masks, sam_objects, area = 50):
    """
    Perform majority voting on prediction masks using SAM segmentation objects.
    Only applies voting to masks with area > 50.
    
    Args:
        prediction_masks (list): List of 2D NumPy arrays with pixel-wise classification predictions.
        sam_objects (list): List of lists, where each inner list contains SAM segmentation dictionaries
                            for an image after refinement using the method in: https://github.com/hifexplo/Masking. 
                            Each dict has 'segmentation' and 'area' keys.
    
    Returns:
        list: List of refined prediction masks after majority voting within segmented objects.
    """
    # Ensure the number of images matches
    if len(prediction_masks) != len(sam_objects):
        raise ValueError("Number of prediction masks must match number of SAM object lists.")
    
    refined_masks = []
    area = 50
    
    # Process each image
    for img_idx, (pred_mask, obj_list) in enumerate(zip(prediction_masks, sam_objects)):
        # Get dimensions of the prediction mask
        pred_height, pred_width = pred_mask.shape
        
        # Initialize the refined mask with the original predictions
        refined_mask = pred_mask.copy()
        
        # Process each segmented object in the image
        for obj in obj_list:
            # Check if area exceeds threshold
            if obj.get('area', 0) <= area:
                continue  # Skip this object if area is <= 50
                
            seg_mask = obj['segmentation']
            seg_height, seg_width = seg_mask.shape
            
            # Check if segmentation mask dimensions match prediction mask
            if seg_height != pred_height or seg_width != pred_width:
                # Resize or crop segmentation mask to match prediction mask
                if seg_height > pred_height or seg_width > pred_width:
                    seg_mask = seg_mask[:pred_height, :pred_width]
                elif seg_height < pred_height or seg_width < pred_width:
                    # Pad with zeros if segmentation mask is smaller
                    seg_mask_padded = np.zeros_like(pred_mask, dtype=np.uint8)
                    seg_mask_padded[:seg_height, :seg_width] = seg_mask
                    seg_mask = seg_mask_padded
            
            # Extract the prediction values within the segmented object
            object_pixels = pred_mask[seg_mask == 1]
            
            if len(object_pixels) > 0:
                # Compute the majority label within the object
                majority_label = mode(object_pixels, keepdims=False).mode
                # Assign the majority label to all pixels in the object
                refined_mask[seg_mask == 1] = majority_label
        
        refined_masks.append(refined_mask)
    
    return refined_masks
 
def load_and_decompress(filename):
    with gzip.open(filename, 'rb') as f:
        compressed_data = pickle.load(f)
    
    decompressed_data = []
    for item in compressed_data:
        # Unpack and trim to the original length before reshaping
        segmentation_unpacked = np.unpackbits(item['segmentation'])[:item['original_length']].reshape(item['shape'])
        decompressed_item = {
            'segmentation': segmentation_unpacked,
            'area': item['area'],
            'bbox': item['bbox'],
            'predicted_iou': item['predicted_iou'],
            'point_coords': item['point_coords'],
            'stability_score': item['stability_score'],
            'crop_box': item['crop_box']
        }
        decompressed_data.append(decompressed_item)
    
    return decompressed_data

def calculate_segmentation_metrics(test_gt, majority_voted_masks, num_classes=5):
    """
    Calculate segmentation metrics including F1-score per class, AA, kappa, and OA.
    
    Args:
        test_gt (list): List of 2D numpy arrays containing ground truth masks
        majority_voted_masks (list): List of 2D numpy arrays containing predicted masks
        num_classes (int): Number of classes (default=5)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Flatten all masks into 1D arrays
    y_true = np.concatenate([gt.flatten() for gt in test_gt])
    y_pred = np.concatenate([pred.flatten() for pred in majority_voted_masks])
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Calculate per-class metrics
    f1_scores = []
    class_accuracies = []
    
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Precision, Recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        class_accuracies.append(tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0)
    
    # Average F1-score across classes
    avg_f1_score = np.mean(f1_scores)
    
    # Average Accuracy (AA)
    average_accuracy = np.mean([acc for acc in class_accuracies if not np.isnan(acc)])
    
    # Overall Accuracy (OA) / Pixel Accuracy
    pixel_accuracy = cm.diagonal().sum() / cm.sum()
    
    # Cohen's Kappa
    n = cm.sum()
    p0 = pixel_accuracy  # observed agreement
    pe = 0  # expected agreement by chance
    
    for i in range(num_classes):
        row_sum = cm[i, :].sum()
        col_sum = cm[:, i].sum()
        pe += (row_sum * col_sum) / (n * n)
    
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0
    
    # Compile results
    results = {
        'f1_scores_per_class': f1_scores,
        'average_f1_score': avg_f1_score,
        'average_accuracy': average_accuracy,
        'kappa': kappa,
        'pixel_accuracy': pixel_accuracy,
        'confusion_matrix': cm
    }
    
    return results


# Define the gain_neighborhood_band function
def gain_neighborhood_band(x_train, band, band_patch, patch):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    
    # Center region
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    
    # Left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    
    # Right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    
    return x_train_band


def visualize_mask(Mask, figsize=(12, 8)):
    """
    Plot a ground truth mask without a color bar on the right, labeling the classes.

    Parameters:
        Mask: NumPy array of shape (H, W), containing class labels.
    """

    # Define class names and corresponding colors
    class_names = ['Background', 'Mesh', 'Steel_Black', 'Steel_Grey', 'HTEL_Grey', 'HTEL_Black']
    colors = ['#000000', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # Example colors

    # Create a colormap and normalization for the mask visualization
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, cmap.N)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # Display the mask
    img = ax.imshow(Mask[:240,:325], cmap=cmap, norm=norm, interpolation='none')
    
    plt.axis('off')
    

def apply_majority_voting(pred_mask, debug=False):
    """
    Applies majority voting to a predicted segmentation mask.
    
    Parameters:
        pred_mask (np.ndarray): Predicted segmentation mask (H, W)
        debug (bool): If True, prints debugging information
    Returns:
        np.ndarray: Corrected mask after majority voting
    """
    # Ensure input is integer type
    if not np.issubdtype(pred_mask.dtype, np.integer):
        print("Warning: Input mask is not integer type, converting...")
        pred_mask = pred_mask.astype(np.int32)
    
    # Label connected components
    labeled_mask, num_labels = label(pred_mask > 0, return_num=True, connectivity=2)  # Changed to connectivity=2
    
    # Preserve original background
    majority_voted_mask = np.zeros_like(pred_mask)
    
    if debug:
        print(f"Number of objects found: {num_labels}")
        print(f"Unique values in input mask: {np.unique(pred_mask)}")
    
    # Process each object
    for obj_label in range(1, num_labels + 1):
        object_pixels = (labeled_mask == obj_label)
        obj_values = pred_mask[object_pixels]
        
        if len(obj_values) > 0:  # Check if object has pixels
            unique_classes, counts = np.unique(obj_values[obj_values > 0], return_counts=True)
            if len(unique_classes) > 0:  # Check if there are non-background values
                majority_class = unique_classes[np.argmax(counts)]
                majority_voted_mask[object_pixels] = majority_class
                
                if debug:
                    print(f"Object {obj_label}:")
                    print(f"Class counts: {dict(zip(unique_classes, counts))}")
                    print(f"Assigned class: {majority_class}")
    
    return majority_voted_mask

# Define feature extraction function
def extract_features_and_labels(hsi_input, mask_input, ignore_classes=[0]):
    X = []
    y = []
    if isinstance(hsi_input, list) and isinstance(mask_input, list):
        hsi_list = hsi_input
        mask_list = mask_input
    else:
        hsi_list = [hsi_input]
        mask_list = [mask_input]
    for hsi, mask in zip(hsi_list, mask_list):
        h, w, c = hsi.shape
        for i in range(h):
            for j in range(w):
                label = mask[i, j]
                if label not in ignore_classes:
                    X.append(hsi[i, j, :].flatten())
                    y.append(label)
    return np.array(X), np.array(y)


def reconstruct_predictions(test_masks, predictions, add_offset=True):
    """
    Reconstruct full prediction images from predictions for non-background pixels.
    
    Parameters:
        test_masks (list of np.ndarray): List of ground truth masks of shape (H, W).
            These masks indicate which pixels are foreground (nonzero) and which are background (0).
        predictions (list or np.ndarray): Model predictions for the non-background pixels,
            arranged in the same order as they were extracted (row-major order).
            (For example, if you stored predictions from your DataLoader, you can concatenate them into a 1D array.)
        add_offset (bool): If True, add 1 to predictions to convert from 0-based class indices (used for training)
            to the original labels (1–5).
            
    Returns:
        List[np.ndarray]: A list of reconstructed prediction masks, one per test image,
            each of shape (H, W) where background pixels remain 0.
    """
    # If predictions is a list, concatenate them into one 1D array.
    if isinstance(predictions, list):
        all_preds = np.concatenate([p.flatten() for p in predictions])
    else:
        all_preds = predictions.flatten()
    
    reconstructed_preds = []
    pred_index = 0  # pointer to traverse all_preds

    for mask in test_masks:
        H, W = mask.shape
        # Initialize an empty prediction image (background = 0)
        pred_img = np.zeros((H, W), dtype=all_preds.dtype)
        # Get the indices (in row-major order) where the mask is non-background.
        non_bg_indices = np.where(mask.flatten() > 0)[0]
        n_pixels = len(non_bg_indices)
        if n_pixels > 0:
            # Extract predictions for these pixels
            preds_for_image = all_preds[pred_index:pred_index+n_pixels]
            if add_offset:
                # Convert back to original label space (i.e. 1–5 instead of 0–4)
                preds_for_image = preds_for_image + 1
            # Place the predictions into a flattened version of the full image
            flat_pred = pred_img.flatten()
            flat_pred[non_bg_indices] = preds_for_image
            pred_img = flat_pred.reshape(H, W)
            pred_index += n_pixels
        reconstructed_preds.append(pred_img)
    
    return reconstructed_preds



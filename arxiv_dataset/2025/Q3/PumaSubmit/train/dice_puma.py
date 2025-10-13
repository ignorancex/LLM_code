import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def calculate_dice_from_masks(mask1, mask2, eps=0.00001):
    """Calculate the DICE score between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice_score = (2 * intersection + eps) / (union + eps)
    return dice_score

def calculate_dice_score_with_masks(tif1, tif2, image_shape, eps=0.00001):
    """Calculate the DICE score between two TIF files using masks."""
    tif1 = np.array(Image.open(tif1).resize(image_shape, Image.NEAREST))
    tif2 = np.array(Image.open(tif2).resize(image_shape, Image.NEAREST))

    # If the ground truth (tif1) has 4 channels, use the first channel
    if tif1.ndim == 3 and tif1.shape[-1] == 4:  # Check if it's 4 channels
        tif1 = tif1[:, :, 0]  # Use the first channel (or modify as needed for your use case)

    # If the predictions (tif2) have multiple channels, use the first channel
    if tif2.ndim == 3 and tif2.shape[-1] > 1:
        tif2 = tif2[:, :, 0]  # Use the first channel (or modify as needed)




    dice_scores = {}
    class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis', 5: 'tissue_necrosis'}

    for category in range(1, 6):
        # Generate binary masks for each class
        mask1 = np.where(tif1 == category, 1, 0)
        mask2 = np.where(tif2 == category, 1, 0)

        # If both masks are empty, perfect match
        if np.sum(mask1) == 0 and np.sum(mask2) == 0:
            dice_score = 1.0
        else:
            dice_score = calculate_dice_from_masks(mask1, mask2, eps)

        dice_scores[class_map[category]] = dice_score

    return dice_scores

def calculate_dice_for_files(ground_truth_file, prediction_file, image_shape):
    """Calculate the DICE scores for a single ground truth and prediction file."""
    dice_scores = calculate_dice_score_with_masks(ground_truth_file, prediction_file, image_shape)

    # Calculate the average DICE score across all classes for this file
    class_scores = [score for score in dice_scores.values() if score is not None]
    average_dice = sum(class_scores) / len(class_scores) if class_scores else 0.0
    dice_scores['average_dice'] = average_dice

    return dice_scores

def compute_dice_scores(gt_folder, pred_folder, image_shape=(1024, 1024)):
    """Read ground truth and prediction TIF images, compute Dice scores for all pairs."""
    # Get sorted lists of ground truth and prediction files
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".tif")])

    # Ensure both folders have the same number of files
    if len(gt_files) != len(pred_files):
        raise ValueError("Mismatch in the number of files between ground truth and predictions.")
    mean_dice = 0
    counter=0
    mean_dice_classes = np.zeros(5)
    # Compute Dice scores for each file pair
    overall_scores = {}
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        # print(f"Processing: GT={gt_file}, Pred={pred_file}")

        # Calculate Dice scores for the pair
        scores = calculate_dice_for_files(gt_path, pred_path, image_shape)
        overall_scores[gt_file] = scores
        mean_dice += overall_scores[gt_file]['average_dice']
        mean_dice_classes[0] += scores['tissue_stroma']
        mean_dice_classes[1] += scores['tissue_blood_vessel']
        mean_dice_classes[2] += scores['tissue_tumor']
        mean_dice_classes[3] += scores['tissue_epidermis']
        mean_dice_classes[4] += scores['tissue_necrosis']


        counter += 1

        # print(f"Dice Scores for {gt_file}: {scores}")

    return overall_scores, mean_dice/counter,mean_dice_classes/counter


def calculate_micro_dice_score_with_masks(gt_folder, pred_folder, image_shape, eps=0.00001):
    """
    Calculate the overall micro DICE score across all classes between two folders of TIF masks.

    Args:
        gt_folder (str): Path to the folder containing ground truth TIF masks.
        pred_folder (str): Path to the folder containing predicted TIF masks.
        image_shape (tuple): Shape to resize the images (height, width).
        eps (float): Small value to avoid division by zero.

    Returns:
        dict: Micro DICE scores for each class and the average micro DICE score.
    """
    class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis',
                 5: 'tissue_necrosis'}
    total_gt_mask = {class_name: [] for class_name in class_map.values()}  # Ground truth masks
    total_pred_mask = {class_name: [] for class_name in class_map.values()}  # Predicted masks

    # Get sorted lists of files
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.tif')])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.tif')])

    # Ensure files in both folders match
    if len(gt_files) != len(pred_files):
        raise ValueError("Ground truth and prediction folders must contain the same number of files.")

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Missing file: {gt_path} or {pred_path}")
            continue

        # Load and preprocess the TIF images
        gt_tif = np.array(Image.open(gt_path).resize(image_shape, Image.NEAREST))
        pred_tif = np.array(Image.open(pred_path).resize(image_shape, Image.NEAREST))

        # If ground truth or prediction has more than one channel, use the first channel
        if gt_tif.ndim == 3 and gt_tif.shape[-1] > 1:
            gt_tif = gt_tif[:, :, 0]
        if pred_tif.ndim == 3 and pred_tif.shape[-1] > 1:
            pred_tif = pred_tif[:, :, 0]

        # Accumulate masks for each class
        for category, class_name in class_map.items():
            gt_mask = np.where(gt_tif == category, 1, 0)
            pred_mask = np.where(pred_tif == category, 1, 0)

            total_gt_mask[class_name].append(gt_mask)
            total_pred_mask[class_name].append(pred_mask)

    # Concatenate all masks for each class
    for class_name in class_map.values():
        if total_gt_mask[class_name] and total_pred_mask[class_name]:  # Avoid empty lists
            total_gt_mask[class_name] = np.concatenate(total_gt_mask[class_name], axis=0)
            total_pred_mask[class_name] = np.concatenate(total_pred_mask[class_name], axis=0)
        else:
            total_gt_mask[class_name] = np.zeros(image_shape)
            total_pred_mask[class_name] = np.zeros(image_shape)

    # Calculate the micro DICE score for each class
    micro_dice_scores = {}
    for class_name in class_map.values():
        mask1 = total_gt_mask[class_name]
        mask2 = total_pred_mask[class_name]

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)

        dice_score = (2 * intersection + eps) / (union + eps)
        if intersection == 0:
            dice_score = 0.0
        if union == 0:
            dice_score = 1.0

        micro_dice_scores[class_name] = dice_score

    # Calculate the average micro DICE score across all classes
    average_dice_score = np.mean(list(micro_dice_scores.values()))
    micro_dice_scores['average_micro_dice'] = average_dice_score

    return average_dice_score, list(micro_dice_scores.values())


# Example Usage
if __name__ == "__main__":
    ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images"
    prediction_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution

    dice_scores = compute_dice_scores(ground_truth_folder, prediction_folder, image_shape)
    # print("Overall Dice Scores:")
    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")

import torch
import tifffile
import numpy as np
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import json
import os
from PIL import Image

def compute_puma_dice_micro_dice_prediction(model = None, image_tensor = None, device = None, weights_list = None):

    # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
    # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

    # Get prediction
    prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list, device=device)


    return prediction

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
    processor_path = "custom_segformer_processor.json"
    with open(processor_path, "r") as f:
        processor_dict = json.load(f)

    processor = SegformerImageProcessor.from_dict(processor_dict)

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
        state_dict = torch.load(weight_path, map_location="cuda:0")
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_probabilities = []
        for i, augmented_input in enumerate(augmented_inputs):
            augmented_input = augmented_input.to(device)
            try:
                model.segformer
                # Process input images
                images = processor(images=[augmented_input[0, 0:3].permute(1, 2, 0).cpu().numpy()],
                                   return_tensors="pt")  # Now it's ready for SegFormer
                images = {key: value.to(model.device) for key, value in images.items()}
                if augmented_input.shape[1] > 3:
                    images['pixel_values'] = torch.concatenate((images['pixel_values'], augmented_input[:, 3].unsqueeze(1)),dim=1)
                pred = model(**images)
                pred = F.interpolate(pred.logits, size=image_tensor.size()[2:],
                                           mode='bilinear', align_corners=False)




            except:
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
    # final_probabilities[5][final_probabilities[5]>0.1] = 1
    # Final prediction (class-wise argmax)
    final_predictions = torch.argmax(final_probabilities, dim=0)

    # Move to CPU and convert to NumPy
    return final_probabilities

def save_segformer_config(num_output_channels, num_input_channels = 3):

    config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    config.num_channels = num_input_channels

    # Modify the configuration
    config.num_labels = num_output_channels
    config.image_size = 1024  # Input image size

    # Save to JSON file
    if num_output_channels == 3:
        config_save_path = "custom_segformer_config" + str(num_output_channels) + ".json"
    else:
        config_save_path = "custom_segformer_nuclei_config" + str(num_output_channels) + ".json"

    with open(config_save_path, "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    print(f"Config saved to {config_save_path}")

    # Load the pretrained processor
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )

    # Modify processor settings
    processor.do_resize = False
    processor.do_rescale = False

    # Save processor to a JSON file
    if num_input_channels == 3:
        processor_save_path = "custom_segformer_processor" + str(num_output_channels) + ".json"
    else:
        processor_save_path = "custom_segformer_processor_nuclei" + str(num_output_channels) + ".json"

    with open(processor_save_path, "w") as f:
        json.dump(processor.to_dict(), f, indent=4)

    print(f"Processor saved to {processor_save_path}")

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

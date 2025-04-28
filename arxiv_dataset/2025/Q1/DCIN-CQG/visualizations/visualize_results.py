import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
import os

# export path to preprocess before running script
from preprocess.dataset_test import TestDataset
from preprocess.color_transfer import open_img

def generate_mask(image_tensor, model):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)  # (1, num_classes, H, W)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    return mask

def visualize_and_save(image, mask, save_path, label_colors):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Resize mask to match image dimensions if necessary
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create an overlay with custom colors
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    for label, color in label_colors.items():
        mask_overlay[mask == label] = color

    # Combine the image and mask overlay
    alpha = 0.4
    combined = image.copy()
    non_zero_mask = mask > 0
    combined[non_zero_mask] = cv2.addWeighted(image[non_zero_mask], 0.5, mask_overlay[non_zero_mask], 0.5, 0)

    # Save the combined image without title or additional plots
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def preprocess_image_with_dataset(input_value, dataset):
    if isinstance(input_value, str):
        for idx in range(len(dataset.images)):
            if dataset.images[idx]['file_name'] in input_value:
                image_tensor, mask = dataset[idx]
                return image_tensor, mask
        raise ValueError("Image not found in dataset.")
    elif isinstance(input_value, int):
        if input_value < len(dataset):
            image_tensor, mask = dataset[input_value]
            return image_tensor, mask
        else:
            raise IndexError("Index out of range.")
    else:
        raise TypeError("Input must be a file path (str) or an index (int).")

def load_model(model_path, backbone):
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=None,
        in_channels=3,
        classes=5,
        activation=None
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model

def main(image_path, model_paths, names, backbone, output_dir, sub_folder, ds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define custom colors for labels
    label_colors = {
        1: (255, 64, 64),
        2: (255, 255, 100),
        3: (128, 128, 255),
        4: (128, 255, 128)
    }

    # Load ground truth image and mask
    dataset_gt = TestDataset(dataset=ds, transform_size=(768, 512), modes=['dcin_full'])
    gt_image_tensor, gt_mask = preprocess_image_with_dataset(image_path, dataset_gt)
    gt_image = open_img(os.path.join(dataset_gt.image_dir, dataset_gt.images[image_path]['file_name']) if isinstance(image_path, int) else image_path)

    # Create output directory
    output_subdir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_subdir, exist_ok=True)

    # Save original image
    original_image_path = os.path.join(output_subdir, "original_image.png")
    cv2.imwrite(original_image_path, cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR))

    # Save ground truth mask
    ground_truth_path = os.path.join(output_subdir, "ground_truth.png")
    visualize_and_save(gt_image, gt_mask, ground_truth_path, label_colors)

    # Display predictions
    modes_list = [[], ['dcin_full']]
    for col, (model_path, name) in enumerate(zip(model_paths, names)):
        model = load_model(model_path, backbone)
        model.to(device)

        for row, modes in enumerate(modes_list):
            # Load dataset for the current mode
            dataset = TestDataset(dataset=ds, transform_size=(768, 512), modes=modes)
            image_tensor, _ = preprocess_image_with_dataset(image_path, dataset)

            # Generate mask
            image_tensor = image_tensor[0].unsqueeze(0).float().to(device)  # Add batch dimension
            mask = generate_mask(image_tensor, model)

            # Save prediction
            mode_name = "No_DCIN" if row == 0 else "DCIN"
            save_path = os.path.join(output_subdir, f"{name}_{mode_name}.png")
            visualize_and_save(gt_image, mask, save_path, label_colors)

    print(f"Visualizations saved to: {output_subdir}")

if __name__ == "__main__":
    # Paths and settings
    image_path = 100  # Image index or image path
    model_paths = [
        "checkpoints/example.ckpt"
    ]
    names = ["Baseline", "Augs", "CQG"]
    backbone = "efficientnet-b2"
    output_dir = "output_images"
    sub_folder = "example"
    ds = 'lq'

    main(image_path, model_paths, names, backbone, output_dir, sub_folder, ds)
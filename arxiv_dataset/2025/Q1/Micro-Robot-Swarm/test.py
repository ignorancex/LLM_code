import os
import torch
import pandas as pd
from shutil import copyfile
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from E5ImageCombination.E5ImageCombination.draw_black_circle import image


class MultiPoolImageCoordinateDataset(Dataset):
    def __init__(self, pool_path, pool_name):
        self.data_info = []

        # Iterate over all pools in the test directory
        # for pool_name in os.listdir(data_dir):

        data_dir_ori = pool_path


        snap_path = os.path.join(data_dir_ori, pool_name)
        noise_path = os.path.join(data_dir_ori, 'noise')
        gt_img_path = os.path.join(data_dir_ori, f'{pool_name}.png')  # Ground truth image

        if os.path.isdir(pool_path):
            noise_files = [f for f in os.listdir(noise_path) if f.endswith('.xlsx')]
            data_gt_files = [f for f in os.listdir(data_dir_ori) if f.endswith('.xlsx')]

            for noise_info_file in noise_files:
                base_name = os.path.splitext(noise_info_file)[0]
                img_file = f'{base_name}.png'
                mask_file = f'{base_name}_mask.png'

                self.data_info.append(
                    (pool_path, snap_path, pool_name, noise_path, noise_info_file, data_gt_files[0], img_file, mask_file, gt_img_path))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pool_path, snap_path, pool_name, noise_path, noise_info_file, data_gt_file, img_file, mask_file, gt_img_path = self.data_info[idx]

        # Load the data
        data = pd.read_excel(os.path.join(noise_path, noise_info_file))
        data_gt = pd.read_excel(os.path.join(pool_path, data_gt_file))

        img_input = Image.open(os.path.join(noise_path, 'img', img_file)).convert('RGB')
        img_mask = Image.open(os.path.join(noise_path, 'mask', mask_file)).convert('L')
        img_gt = Image.open(gt_img_path).convert('RGB')

        noisy_data_noscale = data[['X', 'Y', 'Theta']].values.astype(np.float32)

        # Normalize X and Y coordinates
        data[['X', 'Y']] = data[['X', 'Y']] / [2245.0, 1587.0]
        data_gt[['X', 'Y']] = data_gt[['X', 'Y']] / [2245.0, 1587.0]

        # Get noisy coordinates and theta for all 221 snapshots
        noisy_coords_theta = data[['X', 'Y', 'Theta']].values.astype(np.float32)

        # Target coordinates and theta for all 221 snapshots
        target_coords_theta = data_gt[['X', 'Y', 'Theta']].values.astype(np.float32)

        img_mask_large = img_mask
        img_mask_large_np = torch.tensor(np.array(img_mask_large) / 255.0, dtype=torch.float32).unsqueeze(0)

        # Downsample the input image and mask
        img_input = img_input.resize((256, 256), Image.ANTIALIAS)
        img_mask = img_mask.resize((256, 256), Image.NEAREST)
        img_gt = img_gt.resize((256, 256), Image.NEAREST)

        # Normalize images and convert to tensors
        img_input_tensor = torch.tensor(np.array(img_input).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        img_mask_tensor = torch.tensor(np.array(img_mask) / 255.0, dtype=torch.float32).unsqueeze(0)
        img_gt_tensor = torch.tensor(np.array(img_gt).transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        # Load 221 mask images for snapshots
        masks = []
        for i in range(1, 222):
            snapshot_filename = f"{pool_name}_{i}.png"
            snapshot_path = os.path.join(snap_path, snapshot_filename)
            mask = Image.open(snapshot_path).convert('L').resize((160, 110), Image.NEAREST)
            mask = np.array(mask) / 255.0
            mask[mask < 1] = 0
            masks.append(mask)

        # Stack all 221 masks into a tensor with shape [221, 160, 110]
        masks_tensor = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32)

        # Convert noisy coordinates to a tensor
        noisy_input = torch.tensor(noisy_coords_theta, dtype=torch.float32)

        return img_input_tensor, img_gt_tensor, img_mask_tensor, noisy_input, target_coords_theta, masks_tensor, img_mask_large_np,noisy_data_noscale

# Your existing `CoordinateCorrectionNetwork` and other code here
class CoordinateCorrectionNetwork(nn.Module):
    def __init__(self):
        super(CoordinateCorrectionNetwork, self).__init__()

        ##### Snapshot Branch #####
        self.conv_emb = nn.Sequential(nn.Conv2d(in_channels=221, out_channels=221, kernel_size=3, stride=1, padding=1,groups=221),
            nn.ReLU(True))
        self.conv1_snap =nn.Sequential(
            nn.Conv2d(in_channels=221, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))
        self.conv2_snap = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))
        self.conv3_snap = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        ##### Image Branch #####
        self.conv1_img = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_img = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_img = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        ##### Coordinate Branch #####
        self.fc1_coords = nn.Linear(3, 128)
        self.fc2_coords = nn.Linear(128, 64)
        self.fc3_coords = nn.Linear(64, 16)
        self.fc4_coords = nn.Linear(16, 4)

        ##### Fully Connected Layers #####
        self.fc1_combined = nn.Linear(256 + 256 + 884, 1024)  # Combined features from snapshots, images, and coordinates

        ## for ablation, only coor
        # self.fc1_combined = nn.Linear(256 + 884, 1024)  # Combined features from snapshots, images, and coordinates

        self.fc2_combined = nn.Linear(1024, 2048)
        self.fc3_combined = nn.Linear(2048, 221*3)  # Output for X, Y, Theta

    def forward(self, img_input, masks_tensor, coords_theta):
        ##### Snapshot Branch #####
        masks_tensor = self.conv_emb(masks_tensor)
        x_snap = self.conv1_snap(masks_tensor)
        x_snap = self.conv2_snap(x_snap)
        x_snap = self.conv3_snap(x_snap)
        x_snap = x_snap.mean(dim=[2, 3])  # Squeeze spatial dimensions, resulting in shape [B, 256]

        ##### Image Branch #####
        x_img = F.relu(self.conv1_img(img_input))
        x_img = F.relu(self.conv2_img(x_img))
        x_img = F.relu(self.conv3_img(x_img))
        x_img = x_img.mean(dim=[2, 3])  # Squeeze spatial dimensions, resulting in shape [B, 256]

        ##### Coordinate Branch #####
        x_coords = F.relu(self.fc1_coords(coords_theta))  # Shape [B, 221, 128]
        x_coords = F.relu(self.fc2_coords(x_coords))  # Shape [B, 221, 64]
        x_coords = F.relu(self.fc3_coords(x_coords))  # Shape [B, 221, 16]
        x_coords = F.relu(self.fc4_coords(x_coords))  # Shape [B, 221, 4]
        x_coords = x_coords.view(x_coords.size(0), -1)  # Flatten to shape [B, 221 * 4]

        ##### Combine Branches #####
        x_combined = torch.cat((x_snap, x_img, x_coords), dim=1)  # Combine all features

        #####  for ablation, only snap and coor #####
        # x_combined = torch.cat((x_img, x_coords), dim=1)  # Combine all features

        ### for only coor
        # x_combined =x_coords


        x_combined = F.relu(self.fc1_combined(x_combined))
        x_combined = F.relu(self.fc2_combined(x_combined))
        coords_output = self.fc3_combined(x_combined)  # Predict original coordinates and Theta

        coords_output = coords_output.view(-1, 221, 3)

        return coords_output
def rescale_coordinates(X, Y):
    """Rescale X, Y coordinates back to original dimensions."""
    X_rescaled = X * 2245.0
    Y_rescaled = Y * 1587.0
    return X_rescaled, Y_rescaled


def save_predictions_to_excel(original_df, predictions, output_file):
    """Save predictions to an Excel file, keeping the original format."""
    # Update the DataFrame with the new X, Y, and Theta values
    original_df['X'] = [pred[0] for pred in predictions]
    original_df['Y'] = [pred[1] for pred in predictions]
    original_df['Theta'] = [pred[2] for pred in predictions]

    # Save the updated DataFrame to an Excel file
    original_df.to_excel(output_file, index=False)



# def test_model(data_dir, model, batch_size=1):
def test_model(pool_path, pool_name, model, batch_size=1):
    # Iterate over all pool directories in the test directory
    # for pool_name in os.listdir(data_dir):
    #     pool_path = os.path.join(data_dir, pool_name)


    if os.path.isdir(pool_path):
        print(f"Testing {pool_name}...")

        # Load the dataset
        dataset = MultiPoolImageCoordinateDataset(pool_path, pool_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        original_xlsx_path = os.path.join(pool_path, f'{pool_name}_snapshots_info.xlsx')
        original_df = pd.read_excel(original_xlsx_path)

        all_ground_truth = []
        all_predictions = []
        overlap_pred_gt_list = []
        overlap_noise_gt_list = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                predictions = []
                img_input, img_gt, img_mask, noisy_inputs, targets, masks_tensor, img_mask_large_np, noisy_data_noscale = data
                img_input, img_gt, img_mask, noisy_inputs, targets, masks_tensor, img_mask_large_np, noisy_data_noscale  = img_input.to(device), img_gt.to(device), img_mask.to(device), noisy_inputs.to(device), targets.to(device), masks_tensor.to(device), img_mask_large_np.to(device), noisy_data_noscale.to(device)

                # Forward pass through the model
                # outputs = model(img_mask, masks_tensor, noisy_inputs)

                outputs = model(img_mask, masks_tensor, noisy_inputs)


                # Split the outputs into (X, Y) and Theta
                outputs_xy = outputs[:, :, :2]
                outputs_theta = outputs[:, :, 2]

                # Rescale X and Y
                for batch_idx in range(outputs_xy.size(0)):
                    X_rescaled, Y_rescaled = rescale_coordinates(outputs_xy[batch_idx, :, 0], outputs_xy[batch_idx, :, 1])
                    for j in range(X_rescaled.size(0)):
                        predictions.append([X_rescaled[j].item(), Y_rescaled[j].item(), outputs_theta[batch_idx, j].item()])

                        all_ground_truth.append(targets[batch_idx, j].cpu().numpy())
                        # all_predictions.append(
                        #     [outputs_xy[batch_idx, :, 0], outputs_xy[batch_idx, :, 1], outputs_theta[batch_idx, j].item()])
                        all_predictions.append(
                            [X_rescaled[j].item()/2245.0, Y_rescaled[j].item()/1587.0, outputs_theta[batch_idx, j].item()])



                # predicted_xlsx_file = os.path.join(pool_path, 'output', f'predicted_snapshots_info{i+1}.xlsx')

                output_path = os.path.join(pool_path, 'output_test_ours')
                os.makedirs(output_path, exist_ok=True)

                predicted_xlsx_file = os.path.join(output_path, f'predicted_snapshots_info{i + 1}.xlsx')


                save_predictions_to_excel(original_df, predictions, predicted_xlsx_file)

                print(f"Predictions saved to {predicted_xlsx_file}")


                #################################
                # Combine snapshots for ground truth coordinates
                ground_truth_mask = combine_snapshots_from_xlsx(os.path.join(pool_path, pool_name), original_xlsx_path,
                                                                output_path, iteration=f'gt_{i + 1}')
                # Combine snapshots for predicted coordinates
                predicted_mask = combine_snapshots_from_xlsx(os.path.join(pool_path, pool_name), predicted_xlsx_file,
                                                             output_path, iteration=f'pred_{i + 1}')
                # Combine snapshots for noisy coordinates (using noisy inputs)
                # noise_xlsx_file = os.path.join(output_path, f'noise_snapshots_info_{i + 1}.xlsx')
                # save_predictions_to_excel(original_df, noisy_inputs.cpu().numpy().tolist(), noise_xlsx_file)
                # noise_mask = combine_snapshots_from_xlsx(os.path.join(pool_path, pool_name), noisy_data_noscale,
                #                                          output_path, iteration=f'noise_{i + 1}')

                # Calculate overlap percentages
                overlap_pred_gt = calculate_overlap(ground_truth_mask, predicted_mask)
                overlap_noise_gt = calculate_overlap(ground_truth_mask, img_mask_large_np.cpu().numpy())

                overlap_pred_gt_list.append(overlap_pred_gt)
                overlap_noise_gt_list.append(overlap_noise_gt)

                print(
                    f"Iteration {i + 1}: Overlap (GT vs Pred): {overlap_pred_gt:.2f}%, Overlap (GT vs Noise): {overlap_noise_gt:.2f}%")





                pool_output_dir = os.path.join(output_dir, pool_name)
                os.makedirs(pool_output_dir, exist_ok=True)

                # snap_path = os.path.join(pool_path, pool_name)

                # Combine snapshots and create a mask for overlap calculation
                # overlap_percentage = combine_snapshots_from_xlsx(os.path.join(pool_path, pool_name),
                #                                                  predicted_xlsx_file, output_path, img_mask_large_np,
                #                                                  iteration=i + 1)
                # overlap_percentages.append(overlap_percentage)

                # combine_snapshots_from_xlsx(snap_path, predicted_xlsx_file, pool_output_dir, iteration=i+1)

        # Convert ground truth and predictions to NumPy arrays for metrics calculation
        all_ground_truth = np.array(all_ground_truth)
        all_predictions = np.array(all_predictions)

        # Calculate metrics
        metrics = calculate_metrics(all_ground_truth, all_predictions)

        print(f"Metrics for {pool_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

        # Calculate and display the overall overlap percentages
        avg_overlap_pred_gt = np.mean(overlap_pred_gt_list)
        avg_overlap_noise_gt = np.mean(overlap_noise_gt_list)

        print(f"Overall overlap (GT vs Predicted) for {pool_name}: {avg_overlap_pred_gt:.2f}%")
        print(f"Overall overlap (GT vs Noise) for {pool_name}: {avg_overlap_noise_gt:.2f}%")

    return predicted_xlsx_file, metrics, avg_overlap_pred_gt, avg_overlap_noise_gt



def process_pools(data_dir, model, output_dir):
    all_metrics = []
    all_overlap_pre_gt = []
    all_overlap_noise_gt = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all pool directories in the test directory
    for pool_name in os.listdir(data_dir):
        pool_path = os.path.join(data_dir, pool_name)
        snap_path = os.path.join(pool_path, pool_name)
        if os.path.isdir(pool_path):
            print(f"Processing {pool_name}...")

            # Run the test model and generate predictions
            predicted_xlsx_file, metrics, avg_overlap_pred_gt, avg_overlap_noise_gt = test_model(pool_path, pool_name, model)
            # predicted_xlsx_file = test_model(pool_path, model)
            all_metrics.append(metrics)
            all_overlap_pre_gt.append(avg_overlap_pred_gt)
            all_overlap_noise_gt.append(avg_overlap_noise_gt)


            # Combine snapshots based on the predicted coordinates
            # pool_output_dir = os.path.join(output_dir, pool_name)
            # os.makedirs(pool_output_dir, exist_ok=True)
            #
            # combine_snapshots_from_xlsx(snap_path, predicted_xlsx_file, pool_output_dir)

    # Aggregate metrics across all pools
    aggregate_metrics = {}
    for metric in all_metrics[0].keys():
        aggregate_metrics[metric] = np.mean([m[metric] for m in all_metrics])



    # Display overall metrics
    print("\nOverall Metrics Across All Pools:")
    for metric_name, value in aggregate_metrics.items():
        print(f"{metric_name}: {value}")

        # Calculate the mean overlap percentages across all pools
    mean_overlap_pre_gt = np.mean(all_overlap_pre_gt)
    mean_overlap_noise_gt = np.mean(all_overlap_noise_gt)

    print(f"\nOverall Mean Overlap (GT vs Predicted): {mean_overlap_pre_gt:.2f}%")
    print(f"Overall Mean Overlap (GT vs Noise): {mean_overlap_noise_gt:.2f}%")

def combine_snapshots_from_xlsx(snap_path, xlsx_file, output_dir, iteration):
    """Combine snapshots based on the coordinates and save the combined image."""
    os.makedirs(output_dir, exist_ok=True)

    snapshots_df = pd.read_excel(xlsx_file)
    canvas_size = (2245, 1587)
    canvas = Image.new("RGB", canvas_size)
    dpi = 75

    mask_canvas = Image.new("L", canvas_size, 0)  # Mask to track where snapshots are pasted
    # snapshots_dir = os.path.dirname(xlsx_file)
    snapshots_dir = snap_path
    for index, row in snapshots_df.iterrows():
        snapshot_image_path = os.path.join(snapshots_dir, row['Filename'])
        snapshot_image = Image.open(snapshot_image_path)

        if snapshot_image.mode == "P":
            snapshot_image = snapshot_image.convert("RGBA")

        rotated_snapshot = snapshot_image.rotate(-math.degrees(row['Theta']), expand=True)

        x_pos = int(row['X'] - rotated_snapshot.width // 2)
        y_pos = int(row['Y'] - rotated_snapshot.height // 2)

        canvas.paste(rotated_snapshot, (x_pos, y_pos), rotated_snapshot.convert("RGBA"))

        # Create a mask for the rotated snapshot ###################
        snapshot_mask = Image.new("L", rotated_snapshot.size, 0)
        snapshot_mask.paste(255, (0, 0), rotated_snapshot.split()[3])  # Use the alpha channel to create the mask

        # Paste the mask onto the main mask canvas
        mask_canvas.paste(snapshot_mask, (x_pos, y_pos), snapshot_mask)

    combined_image_path = os.path.join(output_dir, f'combined_image_{iteration}.png')
    combined_mask_path = os.path.join(output_dir, f'combined_mask_{iteration}.png')
    # combined_image_path = os.path.join(output_dir, 'combined_image.png')

    canvas.save(combined_image_path)
    mask_canvas.save(combined_mask_path)

    # Copy the Excel file to the output directory
    # copyfile(xlsx_file, os.path.join(output_dir, os.path.basename(xlsx_file)))

    print(f"Combined image saved to {combined_image_path}")
    print(f"Mask saved to {combined_mask_path}")
    print(f"Excel file saved to {os.path.join(output_dir, os.path.basename(xlsx_file))}")

    # # Calculate overlap between the generated mask and the ground truth mask
    # mask_gt = img_mask_tensor.squeeze().cpu().numpy() * 255  # Convert ground truth mask tensor to NumPy
    # mask_generated = np.array(mask_canvas)  # Convert generated mask to NumPy
    #
    # overlap = np.logical_and(mask_gt, mask_generated).sum()
    # total_area = np.logical_or(mask_gt, mask_generated).sum()
    #
    # overlap_percentage = (overlap / total_area) * 100 if total_area > 0 else 0
    # print(f"Overlap between ground truth mask and generated mask: {overlap_percentage:.2f}%")




    return mask_canvas

def calculate_overlap(mask1, mask2):
    """Calculate the percentage overlap between two masks."""
    mask1_np = np.array(mask1)
    mask2_np = np.array(mask2)

    overlap = np.logical_and(mask1_np, mask2_np).sum()
    total_area = np.logical_or(mask1_np, mask2_np).sum()

    return (overlap / total_area) * 100 if total_area > 0 else 0

def calculate_iou(mask1, mask2):
    """Calculate the Intersection over Union (IoU) between two masks."""
    mask1_np = np.array(mask1)
    mask2_np = np.array(mask2)

    # Calculate Intersection and Union
    intersection = np.logical_and(mask1_np, mask2_np).sum()
    union = np.logical_or(mask1_np, mask2_np).sum()

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou * 100  # Return IoU as a percentage

def calculate_metrics(ground_truth, predictions):
    # Convert ground truth and predictions to NumPy arrays if they are PyTorch tensors
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    """Calculate various metrics for X, Y, and Theta."""
    gt_x, gt_y, gt_theta = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
    pred_x, pred_y, pred_theta = predictions[:, 0], predictions[:, 1], predictions[:, 2]

    # Mean Squared Error (MSE)
    mse_x = mean_squared_error(gt_x, pred_x)
    mse_y = mean_squared_error(gt_y, pred_y)
    mse_theta = mean_squared_error(gt_theta, pred_theta)

    # Root Mean Squared Error (RMSE)
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    rmse_theta = np.sqrt(mse_theta)

    # Mean Absolute Error (MAE)
    mae_x = mean_absolute_error(gt_x, pred_x)
    mae_y = mean_absolute_error(gt_y, pred_y)
    mae_theta = mean_absolute_error(gt_theta, pred_theta)

    # Euclidean Distance for (X, Y)
    euclidean_distances = np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    mean_euclidean_distance = np.mean(euclidean_distances)

    # R-Squared (R2) for X, Y, and Theta
    r2_x = r2_score(gt_x, pred_x)
    r2_y = r2_score(gt_y, pred_y)
    r2_theta = r2_score(gt_theta, pred_theta)

    return {
        "MSE_X": mse_x, "MSE_Y": mse_y, "MSE_Theta": mse_theta,
        "RMSE_X": rmse_x, "RMSE_Y": rmse_y, "RMSE_Theta": rmse_theta,
        "MAE_X": mae_x, "MAE_Y": mae_y, "MAE_Theta": mae_theta,
        "Mean_Euclidean_Distance": mean_euclidean_distance,
        "R2_X": r2_x, "R2_Y": r2_y, "R2_Theta": r2_theta
    }




# Example usage:
# Define the test directory and model
test_data_dir = './test_1000/'
model = CoordinateCorrectionNetwork()  # Load your trained model here
### for noisy stitched image as inp
# model.load_state_dict(torch.load('coordinate_correction_checkpoint_epoch_22.pth'))

###### for img mask as inp
# model.load_state_dict(torch.load('./checkpoints_inp_snap_mask_coor/coordinate_correction_checkpoint_epoch_36.pth'))

###### for only coor as inp
model.load_state_dict(torch.load('./coordinate_correction_checkpoint_epoch_1.pth'))

output_dir = './output_1000/'

# Run the test for all pools
os.makedirs(output_dir, exist_ok=True)
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Run the process for all pools in the test directory and save the outputs
process_pools(test_data_dir, model, output_dir)

# Run the test for all pools in the test directory
# test_model(test_data_dir, model, batch_size=1)

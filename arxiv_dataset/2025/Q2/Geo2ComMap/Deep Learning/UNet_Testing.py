import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from UNet import UNet
from AttentionUNet import AttentionUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=5, out_channels=1).to(device)
# model = UNet(in_channels=5, out_channels=3).to(device)  # For Multiple output U-Net
# model = AttentionUNet(in_channels=5, out_channels=3).to(device)  # For Attention U-Net

model.load_state_dict(torch.load('Weights/UnetTP.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Testing data dirctory
input_folder = f"Testing_data/Input"
gt_folder = f"Testing_data/GT"

# Error save path
error_abs_save_path_TP = f"Errors/AbsErrors_TP.npy"  
error_abs_save_path_PGDir = f"Errors/AbsErrors_TP.npy"  # For Multiple output U-Net and Attention U-Net
error_abs_save_path_RI = f"Errors/AbsErrors_TP.npy"     # For Multiple output U-Net and Attention U-Net

# Check if the input and GT folders exist
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder '{input_folder}' not found!")
if not os.path.exists(gt_folder):
    raise FileNotFoundError(f"Ground truth folder '{gt_folder}' not found!")

# Get all the files in the input and GT folders
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')])
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.npy')])

# Ensure the number of files in the input and GT folders match
assert len(input_files) == len(gt_files), "Input and GT folder files do not match!"

# Initialize lists to store the absolute errors
absolute_errors_TP = []
absolute_errors_PGDir = []  # For Multiple output U-Net and Attention U-Net
absolute_errors_RI = []     # For Multiple output U-Net and Attention U-Net

# Iterate through each file in the input and GT folders
for input_file, gt_file in zip(input_files, gt_files):
    # Load the input and GT data
    input_path = os.path.join(input_folder, input_file)
    gt_path = os.path.join(gt_folder, gt_file)
    
    input_data = np.load(input_path).astype(np.float32)
    gt_data = np.load(gt_path)
    
    """ For Multiple output U-Net and Attention U-Net
    # Extract the ground truth data
    GT_TP = gt_data[:, :, 1] * 10
    GT_PGDir = gt_data[:, :, 0]
    GT_RI = gt_data[:, :, 2] / 10
    """

    # Add a batch dimension and move the input data to the device
    input_data = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0)
    input_data = input_data.to(device)

    # Perform inference
    with torch.no_grad():
        predicted_output = model(input_data).squeeze().cpu().numpy()  # 模型輸出 (轉回 CPU 並轉換為 NumPy)
    
    """ For Multiple output U-Net and Attention U-Net
    predicted_TP = predicted_output[1, :, :] * 10
    predicted_PGDir = predicted_output[0, :, :]
    predicted_RI = predicted_output[2, :, :] / 10
    """
    
    predicted_output = predicted_output * 10
    
    """ For Multiple output U-Net and Attention U-Net
    # Abs error for Tput
    abs_error_TP = np.abs(predicted_TP - GT_TP)
    absolute_errors_TP.extend(abs_error_TP.flatten())  # 儲存展平的絕對誤差
    
    # Abs error for P_dir
    abs_error_PGDir = np.abs(predicted_PGDir - GT_PGDir)
    absolute_errors_PGDir.extend(abs_error_PGDir.flatten())  # 儲存展平的絕對誤差
    
    # Abs error for RI
    abs_error_RI = np.abs(predicted_RI - GT_RI)
    absolute_errors_RI.extend(abs_error_RI.flatten())  # 儲存展平的絕對誤差
    """
    # Abs error for Tput
    abs_error_TP = np.abs(predicted_output - gt_data)
    absolute_errors_TP.extend(abs_error_TP.flatten())  
    
# Save the absolute errors to a file
np.save(error_abs_save_path_TP, np.array(absolute_errors_TP))
print(f"Abs error saved at {error_abs_save_path_TP}")

""" For Multiple output U-Net and Attention U-Net   
# Save the absolute errors to a file
np.save(error_abs_save_path_TP, np.array(absolute_errors_TP))
print(f"Abs error saved at {error_abs_save_path_TP}")

np.save(error_abs_save_path_PGDir, np.array(absolute_errors_PGDir))
print(f"Abs error saved at {error_abs_save_path_PGDir}")

np.save(error_abs_save_path_RI, np.array(absolute_errors_RI))
print(f"Abs error saved at {error_abs_save_path_RI}")
"""
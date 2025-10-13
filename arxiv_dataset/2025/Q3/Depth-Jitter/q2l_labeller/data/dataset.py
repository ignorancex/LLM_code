import cv2
import torch
import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, resize

def compute_I_without_J(I_orig, beta, z, z_mod, B, gamma):
    a = torch.exp(-beta * z)
    b = torch.exp(-beta * z_mod)
    exp_gamma_z_mod = torch.exp(-gamma * z_mod)
    exp_gamma_z = torch.exp(-gamma * z)
    I_mod = (I_orig * b + B * (a * (1 - exp_gamma_z_mod) - b * (1 - exp_gamma_z))) / a
    return I_mod

class SeaThruAugmentation:
    def __init__(self, image_folder, depth_image_folder, depth_npy_folder,seathru_parameters_path,depth_variance_path, mode='train', resize=(384, 384), threshold=9.49,dataset='FathomNet'):
        """
        Args:
            - `image_folder`: Path to original images.
            - `depth_image_folder`: Path to depth images.
            - `depth_npy_folder`: Path to depth npy files.
            - `seathru_parameters_path`: Path to JSON file with Sea-thru parameters.
            - `depth_variance_path`: Path to JSON file containing precomputed depth variances.
            - `threshold`: Depth variance threshold (default: 9.49).
        """
        # self.mean_depth = depth_stats["mean_depth"]
        # self.std_depth = depth_stats["std_depth"]
        # self.p10 = depth_stats["p10"]
        # self.p90 = depth_stats["p90"]
        self.image_folder = Path(image_folder)
        self.depth_image_folder = Path(depth_image_folder)
        self.depth_npy_folder = Path(depth_npy_folder)
        self.seathru_parameters_path = seathru_parameters_path
        self.depth_variance_path = depth_variance_path
        self.dataset = dataset
        self.threshold = threshold
        self.mode = mode
        self.image_files = list(self.image_folder.glob('*.jpg'))
        self.resize = resize
        
        with open(seathru_parameters_path, 'r') as file:
            self.data = json.load(file)  # Sea-thru parameters
        
        with open(depth_variance_path, 'r') as file:
            self.variances = json.load(file)  # Depth variances

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_tensor_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.resize)

    def __call__(self, image_name, image):
        depth_image_file = self.depth_image_folder / (Path(image_name).stem + '.png')  # UTDAC Dataset
        depth_npy_file = self.depth_npy_folder / (Path(image_name).stem + '_raw_depth_meter.npy')

        if not depth_image_file.exists() or not depth_npy_file.exists():
            raise FileNotFoundError(f"Depth files not found for {image_name}. Skipping.")

        # Get depth variance
        variance = self.variances.get(image_name, 0)

        # If variance is too low, return original image (skip augmentation)
        if variance <= self.threshold:
            return self.preprocess_image(image)

        # Otherwise, apply augmentation
        return self.compute_augmentation(image_name, image, str(depth_image_file), str(depth_npy_file))

    def preprocess_image(self, image):
        """ Convert the image to tensor and resize before returning (no augmentation). """
        image = resize(Image.fromarray(image), self.resize)
        return to_tensor(image)

    def compute_augmentation(self, image_name, image, depth_image_path, depth_npy_path):
        """ Apply depth-based augmentation if variance is above the threshold. """

        # Resize the image first
        image = resize(Image.fromarray(image), self.resize)
        image = np.array(image)

        # Load the depth image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        depth_image = cv2.resize(depth_image, (self.resize[1], self.resize[0]))  # Resize to match image
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        distance_map_gray_image = torch.mean(depth_image, dim=2)

        # Load the depth data from the npy file
        depth_data = np.load(depth_npy_path)
        depth_data_resized = cv2.resize(depth_data, (self.resize[1], self.resize[0]))  # Resize to match image
        distance_map_gray_npy = torch.tensor(depth_data_resized, dtype=torch.float32)

        # Ensure both depth maps are the same shape
        if distance_map_gray_image.shape != distance_map_gray_npy.shape:
            raise ValueError("Depth image and metric depth npy file must have the same shape")

        # Select pixels where depth exists in both sources
        v_valid, u_valid = torch.where((distance_map_gray_image > 0) & (distance_map_gray_npy > 0))
        image_valid = torch.tensor(image, dtype=torch.float32)[v_valid, u_valid]
        distance_map_valid_npy = distance_map_gray_npy[v_valid, u_valid]

        # **Random Depth Offset Augmentation** (only applied when variance > threshold)
        # depth_offset = random.uniform(-2, 6)
        # depth_offset = random.uniform(-self.std_depth, self.std_depth * 2)
        depth_min = np.min(depth_data[depth_data > 0])  # Smallest valid depth
        depth_max = np.max(depth_data)
        # depth_offset = random.uniform(-0.8 * depth_min, 0.5 * depth_max)
        depth_offset = random.uniform(-5,5)

        random_B = random.uniform(0, 0.4)
        random_betac = random.uniform(0, 0.04)

        # Retrieve parameters
        betac = [self.data[image_name]['channel_0']['betac'], self.data[image_name]['channel_1']['betac'], self.data[image_name]['channel_2']['betac']]
        gammac = [self.data[image_name]['channel_0']['gammac'], self.data[image_name]['channel_1']['gammac'], self.data[image_name]['channel_2']['gammac']]
        Bc = [self.data[image_name]['channel_0']['Bc'], self.data[image_name]['channel_1']['Bc'], self.data[image_name]['channel_2']['Bc']]

        for i in range(len(Bc)):
            if Bc[i] == 1:
                Bc[i] -= random.uniform(0, 0.4)

        # Compute the modified image using depth npy
        I_mod_npy = torch.full(image.shape, torch.nan, dtype=torch.float32)
        for channel in range(3):
            I_mod_npy[v_valid, u_valid, channel] = compute_I_without_J(
                image_valid[:, channel], betac[channel] + random_betac, distance_map_valid_npy, distance_map_valid_npy + depth_offset, Bc[channel], gammac[channel]
            )

        # Replace NaNs with 0
        I_mod_npy = torch.nan_to_num(I_mod_npy)

        # Convert to tensor
        restored_image_npy = I_mod_npy.cpu().numpy().clip(0, 255).astype('uint8')
        restored_image_npy = Image.fromarray(restored_image_npy)
        restored_image_npy = self.resize_transform(restored_image_npy)
        restored_image_npy = to_tensor(restored_image_npy)

        if len(restored_image_npy.shape) == 3 and restored_image_npy.shape[0] != 3:
            restored_image_npy = restored_image_npy.permute(2, 0, 1)

        return restored_image_npy
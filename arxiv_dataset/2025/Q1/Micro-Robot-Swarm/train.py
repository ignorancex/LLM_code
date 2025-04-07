import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="stitching",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "noisy-snapshots",
    "epochs": 10,
    }
)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clear CUDA cache before starting training
torch.cuda.empty_cache()


def combine_and_save_images(input_image, ground_truth, generated_image, epoch, output_dir):
    """Combine input, ground truth, and generated images into one and save it."""
    combined_image = torch.cat((input_image, ground_truth, generated_image), dim=2)  # Concatenate images along width
    save_path = os.path.join(output_dir, f'epoch_{epoch + 1}.png')
    save_image(combined_image, save_path)
    print(f'Saved combined image at epoch {epoch + 1}: {save_path}')

class MultiPoolImageCoordinateDataset(Dataset):
    def __init__(self, data_dir):
        self.data_info = []  # List to hold information for each pool

        # Iterate over all pool directories and collect data information
        for pool_name in os.listdir(data_dir):
            pool_path = os.path.join(data_dir, pool_name)
            noise_path = os.path.join(pool_path, 'noise')
            gt_img_path = os.path.join(pool_path, f'{pool_name}.png')  # Load ground truth image

            if os.path.isdir(pool_path):
                noise_files = [f for f in os.listdir(noise_path) if f.endswith('.xlsx')]
                data_gt_files = [f for f in os.listdir(pool_path) if f.endswith('.xlsx')]

                for noise_info_file in noise_files:
                    base_name = os.path.splitext(noise_info_file)[0]
                    img_file = f'{base_name}.png'
                    mask_file = f'{base_name}_mask.png'

                    self.data_info.append(
                        (pool_path, pool_name, noise_path, noise_info_file, data_gt_files[0], img_file, mask_file, gt_img_path))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pool_path, pool_name,noise_path, noise_info_file, data_gt_file, img_file, mask_file, gt_img_path = self.data_info[idx]

        # Load the data
        data = pd.read_excel(os.path.join(noise_path, noise_info_file))
        data_gt = pd.read_excel(os.path.join(pool_path, data_gt_file))

        img_input = Image.open(os.path.join(noise_path, 'img', img_file)).convert('RGB')
        img_mask = Image.open(os.path.join(noise_path, 'mask', mask_file)).convert('L')
        img_gt = Image.open(gt_img_path).convert('RGB')

        # Normalize X and Y coordinates
        data[['X', 'Y']] = data[['X', 'Y']] / [2245.0, 1587.0]
        data_gt[['X', 'Y']] = data_gt[['X', 'Y']] / [2245.0, 1587.0]

        # Get noisy coordinates and theta for all 221 snapshots
        noisy_coords_theta = data[['X', 'Y', 'Theta']].values.astype(np.float32)

        # Target coordinates and theta for all 221 snapshots
        target_coords_theta = data_gt[['X', 'Y', 'Theta']].values.astype(np.float32)

        # Downsample the input image and mask
        img_input = img_input.resize((256, 256), Image.ANTIALIAS)
        img_mask = img_mask.resize((256, 256), Image.NEAREST)
        img_gt = img_gt.resize((256, 256), Image.NEAREST)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(img_input)
        # plt.axis('off')
        # plt.show()

        # Normalize images and convert to tensors
        img_input_tensor = torch.tensor(np.array(img_input).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        img_mask_tensor = torch.tensor(np.array(img_mask) / 255.0, dtype=torch.float32).unsqueeze(0)
        img_gt_tensor = torch.tensor(np.array(img_gt).transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        # Convert noisy and target coordinates to tensors
        noisy_input = torch.tensor(noisy_coords_theta, dtype=torch.float32)
        target = torch.tensor(target_coords_theta, dtype=torch.float32)

        # Initialize an empty list to hold the mask images
        masks = []

        for i in range(1, 222):  # Assuming the files are named pool1_1.png, pool1_2.png, ..., pool1_221.png
            snapshot_filename = f"{pool_name}_{i}.png"
            snapshot_path = os.path.join(pool_path, pool_name, snapshot_filename)

            # Load the snapshot as a binary mask (convert to grayscale and binarize)
            mask = Image.open(snapshot_path).convert('L').resize((160, 110), Image.NEAREST)
            mask = np.array(mask) / 255.0  # Normalize to binary [0, 1]
            mask[mask < 1] = 0
            masks.append(mask)

        # Stack all 221 masks into a tensor with shape [221, 160, 110]
        masks_tensor = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32)

        # Convert noisy coordinates to a tensor
        noisy_input = torch.tensor(noisy_coords_theta, dtype=torch.float32)  # Shape [221, 3]

        return img_input_tensor, img_gt_tensor, img_mask_tensor, noisy_input, target, masks_tensor


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
        # self.fc1_combined = nn.Linear(884,1024)  # Combined features from snapshots, images, and coordinates


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


        x_combined = F.relu(self.fc1_combined(x_combined))
        x_combined = F.relu(self.fc2_combined(x_combined))
        coords_output = self.fc3_combined(x_combined)  # Predict original coordinates and Theta

        coords_output = coords_output.view(-1, 221, 3)

        return coords_output


def train_model(data_dir, epochs=100, batch_size=24, learning_rate=0.001, save_interval=1):
    model = CoordinateCorrectionNetwork().to(device)
    criterion_xy = nn.MSELoss()  # Loss for X, Y coordinates
    criterion_theta = nn.MSELoss()  # Loss for Theta
    criterion_image = nn.MSELoss()  # Loss for image generation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, criterion_xy, log="all", log_freq=10)

    dataset = MultiPoolImageCoordinateDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    output_dir = "./output_images/"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory for images

    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        running_loss_xy = 0.0
        running_loss_theta = 0.0
        running_loss_image = 0.0
        count = 0

        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for i, data in enumerate(dataloader, 0):
                count += 1
                img_input, img_gt, img_mask, noisy_inputs, targets, masks_tensor = data
                img_input, img_gt, img_mask, noisy_inputs, targets, masks_tensor = img_input.to(device), img_gt.to(device), img_mask.to(device), noisy_inputs.to(device), targets.to(device), masks_tensor.to(device)

                optimizer.zero_grad()


                coords_output = model(img_mask, masks_tensor, noisy_inputs)



                # Split the outputs and targets into (X, Y) and Theta
                outputs_xy = coords_output[:, :, :2]
                outputs_theta = coords_output[:, :, 2]
                targets_xy = targets[:, :, :2]
                targets_theta = targets[:, :, 2]

                # Compute losses
                loss_xy = criterion_xy(outputs_xy, targets_xy)
                loss_theta = criterion_theta(outputs_theta, targets_theta)
                # loss_image = criterion_image(img_output, img_gt)  # Image reconstruction loss

                # Combine losses
                # loss = loss_xy + 0.05 * loss_theta + loss_image
                loss = loss_xy + 0.05 * loss_theta

                loss.backward()
                optimizer.step()

                running_loss_xy += loss_xy.item()
                running_loss_theta += loss_theta.item()
                # running_loss_image += loss_image.item()

                if epoch % 2 == 0:
                    # Log the losses to WandB
                    wandb.log({
                        "Loss_XY": loss_xy.item(),
                        "Loss_Theta": loss_theta.item(),
                        # "Loss_Image": loss_image.item(),
                        "Total_Loss": loss.item(),
                        "Epoch": epoch + 1
                    })

                if i % 100 == 99:
                    pbar.set_postfix({
                        'Loss_XY': running_loss_xy / 100,
                        'Loss_Theta': running_loss_theta / 100,
                        'Loss_Image': running_loss_image / 100,
                    })
                    running_loss_xy = 0.0
                    running_loss_theta = 0.0
                    running_loss_image = 0.0


                pbar.update(1)

        # Save the model checkpoint every 100 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint_path = f'coordinate_correction_checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')



    print('Finished Training')

    torch.save(model.state_dict(), 'coordinate_correction_model.pth')
    wandb.finish()



def sample_data(loader):
    while True:
        for batch in loader:
            yield batch




# Define the data directory
data_dir = './data_100/'

# Train the model
train_model(data_dir)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from UNet import UNet  # Ensure UNet is properly imported from your implementation file
from AttentionUNet import AttentionUNet  # Ensure AttentionUNet is properly imported from your implementation file

# Check if CUDA is available
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f'Number of available CUDA devices: {num_devices}')
    for device_id in range(num_devices):
        device_name = torch.cuda.get_device_name(device_id)
        device_capability = torch.cuda.get_device_capability(device_id)
        print(f'Device {device_id}: {device_name}')
        print(f' - Compute Capability: {device_capability}')
        print(f' - Memory Allocated: {torch.cuda.memory_allocated(device_id)} bytes')
        print(f' - Memory Cached: {torch.cuda.memory_reserved(device_id)} bytes')
else:
    print('CUDA is not available on this system.')

# Define the dataset class
class NpyDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_files = sorted(os.listdir(input_dir))
        self.output_files = sorted(os.listdir(output_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])
        
        input_image = np.load(input_path).astype(np.float32)
        output_image = np.load(output_path).astype(np.float32)
        output_image = output_image / 10  # Data preprocess for output
        
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
        output_image = torch.from_numpy(output_image).unsqueeze(0)  # Add channel dimension
        
        return input_image, output_image


""" Dataset and DataLoader for multiple-output U-Net model   
class NpyDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_files = sorted(os.listdir(input_dir))
        self.output_files = sorted(os.listdir(output_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])
        
        input_image = np.load(input_path).astype(np.float32)
        output_image = np.load(output_path).astype(np.float32)
        
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
        output_image = torch.from_numpy(output_image).permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
        
        return input_image, output_image
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

# Define dataset directories
input_dir = 'Training_data/Input'
output_dir = 'Training_data/GT'
validation_input_dir = 'Validation_data/Input'
validation_gt_dir = 'Validation_data/GT'

# Hyperparameters
batch_size = 64
num_epochs = 50
learning_rate = 1e-3

# Model save paths
model_save_path = 'Weights/UnetTP.pth'

# Dataset and DataLoader
train_dataset = NpyDataset(input_dir, output_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NpyDataset(validation_input_dir, validation_gt_dir)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Print dataset lengths
print(f'Training dataset length: {len(train_dataset)}')
print(f'Validation dataset length: {len(val_dataset)}')

# Model, loss, optimizer, and scheduler
model = UNet(in_channels=5, out_channels=1).to(device)

# For Attention U-Net
# model = AttentionUNet(in_channels=5, out_channels=1).to(device)

# For multiple-output U-Net model
# model = UNet(in_channels=5, out_channels=3).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Training and validation
best_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}')

    # Validation
    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item() * val_inputs.size(0)

    epoch_val_loss = val_running_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}')

    # Adjust learning rate
    scheduler.step(epoch_val_loss)

    # Save the model with the best validation loss
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f'Best model saved with Validation Loss: {best_loss:.4f}')

    # Save latest model
    torch.save(model.state_dict(), model_save_path)

print('Training complete.')

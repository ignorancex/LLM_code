import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from data_modules.xray_dataset import XrayDataset
from models.xray_resnet import XrayResNetWithCrossStitch  # Updated model with cross-stitch units
from trainers.mtl_cross_stitch_train import MTLTrainerWithCrossStitch
from testers.mtl_test import MTLTester
import pandas as pd
from tqdm import tqdm
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a cross stitch multi-task learning (MTL) model with Cross-Stitch Networks")
parser.add_argument('--data_dir', type=str, default='/media/alexosama/data/youssefmohamed/image_search/images', help="Directory for image data")
parser.add_argument('--img_size', type=int, default=64, help="Size of the image")
parser.add_argument('--styles_csv_path', type=str, default="/media/alexosama/data/youssefmohamed/image_search/datasets/styles.csv", help="Path to styles CSV file")
parser.add_argument('--dataset_name', type=str, default="xray", choices=["fashion", "botit", "home_office", "xray"], help="Dataset name")
parser.add_argument('--model_name', type=str, default="ResNet", choices=["ResNet"], help="Model name for training")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--save_model_path', type=str, required=True, help="Path to save the trained model")
args = parser.parse_args()

# Parameters
data_dir = args.data_dir
img_size = args.img_size
styles_csv_path = args.styles_csv_path
dataset_name = args.dataset_name
model_name = args.model_name
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
save_model_path = args.save_model_path

# Define dataset registry
dataset_registry = {
    "xray": lambda df, data_dir, img_size, transform: XrayDataset(df, data_dir, img_size, transform)
}

# Define model registry
model_registry = {
    "xray": {
        "ResNet": lambda img_size, df: XrayResNetWithCrossStitch(img_size, df)  # Cross-Stitch-enabled ResNet
    }
}

# Validate dataset and model names
if dataset_name not in dataset_registry:
    raise ValueError(f"Invalid dataset name '{dataset_name}'. Supported datasets: {list(dataset_registry.keys())}.")
if model_name not in model_registry[dataset_name]:
    raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'. Supported models: {list(model_registry[dataset_name].keys())}.")

# Load the dataset
df = pd.read_csv(styles_csv_path, on_bad_lines='skip')
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
full_dataset = dataset_registry[dataset_name](df, data_dir, img_size, transform)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = model_registry[dataset_name][model_name](img_size, df)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize and train the model using the cross-stitch trainer
trainer = MTLTrainerWithCrossStitch(model, train_loader, val_loader, optimizer, criterion, project_name="MTL_CrossStitch")
trainer.train(num_epochs=num_epochs, model_path=save_model_path)

# Test the model
tester = MTLTester(model, test_loader, criterion, "MTL_CrossStitch_Test")
tester.test()

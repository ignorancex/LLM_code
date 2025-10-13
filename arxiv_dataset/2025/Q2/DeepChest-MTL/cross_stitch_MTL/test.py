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
from testers.mtl_test import MTLTester
import pandas as pd
from tqdm import tqdm
import random



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


def load_model(model_name, dataset_name, img_size, df, model_path):
    if dataset_name not in model_registry:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if model_name not in model_registry[dataset_name]:
        raise ValueError(f"Unsupported model: {model_name} for dataset {dataset_name}")
    print(os.path.getsize(model_path))
    print(model_path)
    # model = model_registry[dataset_name][model_name](img_size, df)
    # model.load_state_dict(torch.load(model_path, weights_only=True)) #, weights_only=True
    state_dict = torch.load(model_path)

    exit()
    # Instantiate the model
    model = model_registry[dataset_name][model_name](img_size, df)
    model.load_state_dict(torch.load(model_path)) #, weights_only=True
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Test cross stitch multitask learning model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory for image data")
    parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_registry.keys(), help="The name of the dataset")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model to use")
    parser.add_argument('--styles_csv_path', type=str, required=True, help="Path to styles CSV file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument('--img_size', type=int, default=128, help="Size of the images")
    args = parser.parse_args()

    # Load the dataset CSV
    df = pd.read_csv(args.styles_csv_path, on_bad_lines='skip')

    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the dataset using the registry
    if args.dataset_name not in dataset_registry:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    dataset = dataset_registry[args.dataset_name](df, args.data_dir, args.img_size, transform)

    # Create DataLoader
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = load_model(args.model_name, args.dataset_name, args.img_size, df, args.model_path)

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()

    # Instantiate the tester and run the test
    tester = MTLTester(model, test_loader, criterion)
    tester.test()

if __name__ == "__main__":
    main()


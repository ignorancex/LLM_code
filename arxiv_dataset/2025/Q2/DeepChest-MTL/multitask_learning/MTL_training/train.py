import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from data_modules.fashion_dataset import FashionDataset
from data_modules.botit_dataset import BotitDataset
from data_modules.office_home_dataset import OfficeHomeDataset
from data_modules.xray_dataset import XrayDataset
from models.cnn_model1 import CNNModel1
from models.botit_cnn_model1 import BotitCNNModel1
from models.botit_vgg16 import BotitVGG16
from models.fashion_vgg16 import FashionVGG16
from models.botit_resnet import BotitResNet
from models.fashion_resnet import FashionResNet
from models.fashion_efficientnet import FashionEfficientNet
from models.botit_efficientnet import BotitEfficientNet
from models.botit_densenet import BotitDenseNet
from models.fashion_densenet import FashionDenseNet
from models.botit_mobilenet import BotitMobileNet
from models.fashion_mobilenet import FashionMobileNet
from models.botit_inception import BotitInceptionNet
from models.fashion_inception import FashionInceptionNet
from models.botit_vit import BotitViT
from models.fashion_vit import FashionViT
from models.Botit_swing import BotitSwinTransformer
from models.fashion_swing import FashionSwinTransformer
from models.xray_cnn_model1 import XrayCNNModel1
from models.xray_vgg16 import XrayVGG16
from models.xray_resnet import XrayResNet
from models.xray_inception import XrayInceptionNet
from models.xray_efficientnet import XrayEfficientNet
from models.xray_mobilenet import XrayMobileNet
from models.xray_densenet import XrayDenseNet
from models.xray_vit import XrayViT
from models.xray_swing import XraySwinTransformer
from trainers.mtl_trainer import MTLTrainer
from trainers.mtl_trainer_cw import MTLTrainerCW
from trainers.mtl_dwbstl_train import MTLTrainerWithDWBSTL
from trainers.mtl_dwbstl_cw_train import MTLTrainerWithDWBSTLCW
from testers.mtl_test import MTLTester
import cv2
import pandas as pd
from tqdm import tqdm
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a fashion image classifier using MTL")
parser.add_argument('--data_dir', type=str, default='/media/alexosama/data/youssefmohamed/image_search/images', help="Directory for image data")
parser.add_argument('--img_size', type=int, default=64, help="Size of the image")
parser.add_argument('--styles_csv_path', type=str, default="/media/alexosama/data/youssefmohamed/image_search/datasets/styles.csv", help="Path to styles CSV file")
parser.add_argument('--dataset_name', type=str, default="botit", choices=["fashion", "botit", "home_office", "xray"], help="the name of the used dataset for training")
parser.add_argument('--training_type', type=str, default="MTL-DWBSTL", choices=["MTL-w1", "MTL-DWBSTL"], help="Type of training")
parser.add_argument('--model_name', type=str, default="CNN_model1", choices=["CNN_model1", "VGG16", "ResNet", "efficientnet", "densenet", "mobilenet", "inception", "ViT", "swing"], help="training model name")
parser.add_argument('--class_weights_type', type=str, default="NoN", choices=["NoN", "normal"], help="Type of classes weights for each class")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--initial_weights_config_path', type=str, required=True, help="Path to the JSON config file with initial weights")
parser.add_argument('--save_model_path', type=str, required=True, help="The path to save the trained model")
args = parser.parse_args()

# Parameters
data_dir = args.data_dir
img_size = args.img_size
styles_csv_path = args.styles_csv_path
dataset_name = args.dataset_name
training_type = args.training_type
class_weights_type = args.class_weights_type
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr 
save_model_path = args.save_model_path
initial_weights_config_path = args.initial_weights_config_path
model_name = args.model_name

# Define supported datasets
dataset_registry = {
    "fashion": lambda df, data_dir, img_size, transform: FashionDataset(df, data_dir, img_size, transform),
    "botit": lambda df, data_dir, img_size, transform: BotitDataset(df, data_dir, img_size, transform),
    "home_office": lambda df, data_dir, img_size, transform: OfficeHomeDataset(df, data_dir, img_size, transform),
    "xray": lambda df, data_dir, img_size, transform: XrayDataset(df, data_dir, img_size, transform)
}

# Define supported datasets and models
model_registry = {
    "fashion": {
        "CNN_model1": lambda img_size, df: CNNModel1(img_size, df),
        "VGG16": lambda img_size, df: FashionVGG16(img_size, df),
        "ResNet": lambda img_size, df: FashionResNet(img_size, df),
        "efficientnet": lambda img_size, df: FashionEfficientNet(img_size, df),
        "densenet": lambda img_size, df: FashionDenseNet(img_size, df),
        "mobilenet": lambda img_size, df: FashionMobileNet(img_size, df),
        "inception": lambda img_size, df: FashionInceptionNet(img_size, df),
        "ViT": lambda img_size, df: FashionViT(img_size, df),
        "swing": lambda img_size, df: FashionSwinTransformer(img_size, df)
    },
    "botit": {
        "CNN_model1": lambda img_size, df: BotitCNNModel1(img_size, df),
        "VGG16": lambda img_size, df: BotitVGG16(img_size, df),
        "ResNet": lambda img_size, df: BotitResNet(img_size, df),
        "efficientnet": lambda img_size, df: BotitEfficientNet(img_size, df),
        "densenet": lambda img_size, df: BotitDenseNet(img_size, df),
        "mobilenet": lambda img_size, df: BotitMobileNet(img_size, df),
        "inception": lambda img_size, df: BotitInceptionNet(img_size, df),
        "ViT": lambda img_size, df: BotitViT(img_size, df),
        "swing": lambda img_size, df: BotitSwinTransformer(img_size, df)
    },
    "xray":{
        "CNN_model1": lambda img_size, df: XrayCNNModel1(img_size, df),
        "VGG16": lambda img_size, df: XrayVGG16(img_size, df),
        "ResNet": lambda img_size, df: XrayResNet(img_size, df),
        "efficientnet": lambda img_size, df: XrayEfficientNet(img_size, df),
        "inception": lambda img_size, df: XrayInceptionNet(img_size, df),
        "mobilenet": lambda img_size, df: XrayMobileNet(img_size, df),
        "densenet": lambda img_size, df: XrayDenseNet(img_size, df),
        "ViT": lambda img_size, df: XrayViT(img_size, df),
        "swing": lambda img_size, df: XraySwinTransformer(img_size, df)
    }
}

# Define trainer mappings
trainer_registry = {
    "MTL-w1": {
        "normal": lambda: MTLTrainerCW(model, train_loader, val_loader, optimizer, criterion, "MTL-w1"),
        "default": lambda: MTLTrainer(model, train_loader, val_loader, optimizer, criterion, "MTL-w1")
    },
    "MTL-DWBSTL": {
        "normal": lambda: MTLTrainerWithDWBSTLCW(model, train_loader, val_loader, optimizer, criterion, initial_weights),
        "default": lambda: MTLTrainerWithDWBSTL(model, train_loader, val_loader, optimizer, criterion, initial_weights)
    }
}

# Load initial weights from the config file
with open(initial_weights_config_path, 'r') as f:
    config = json.load(f)
initial_weights = config["initial_weights"]

# Load the dataset
df = pd.read_csv(styles_csv_path, on_bad_lines='skip')
#df = df[:10000]  # Limit to first 10,000 rows

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



# Validate dataset name
if dataset_name not in dataset_registry:
    raise ValueError(f"Invalid dataset name '{dataset_name}'. Supported datasets: {list(dataset_registry.keys())}.")

# Create the dataset
full_dataset = dataset_registry[dataset_name](df, data_dir, img_size, transform)

# Define dataset sizes for splits (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Validate dataset and model names
if dataset_name not in model_registry:
    raise ValueError(f"Invalid dataset name '{dataset_name}'. Supported datasets: {list(model_registry.keys())}.")
if model_name not in model_registry[dataset_name]:
    raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'. Supported models: {list(model_registry[dataset_name].keys())}.")

# Initialize the model
model = model_registry[dataset_name][model_name](img_size, df)


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr)


# Validate training type and class weights
if training_type not in trainer_registry:
    raise ValueError(f"Invalid training type '{training_type}'. Supported types: {list(trainer_registry.keys())}.")
if class_weights_type not in trainer_registry[training_type]:
    class_weights_type = "default"  # Use "default" if class_weights_type is not explicitly handled.

# Initialize and train
trainer = trainer_registry[training_type][class_weights_type]()
trainer.train(num_epochs=num_epochs, model_path=save_model_path)


# Instantiate and run the tester
tester = MTLTester(model, test_loader, criterion, "first mtl test")
tester.test()

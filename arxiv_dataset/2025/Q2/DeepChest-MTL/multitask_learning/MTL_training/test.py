import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
# from models.cnn_model1 import CNNModel1  # Ensure you import your model here
# from models.botit_cnn_model1 import BotitCNNModel1
# from data_modules.fashion_dataset import FashionDataset  # Ensure your dataset class is correctly imported
# from data_modules.botit_dataset import BotitDataset
# from data_modules.xray_dataset import XrayDataset
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
from models.xray_efficientnet import XrayEfficientNet
from models.xray_inception import XrayInceptionNet
from models.xray_mobilenet import XrayMobileNet
from models.xray_densenet import XrayDenseNet
from models.xray_vit import XrayViT
from models.xray_swing import XraySwinTransformer
from testers.mtl_test import MTLTester  # Ensure you import your tester class
import torch.nn as nn
import torchvision.transforms as transforms

# def load_model(model_path, num_tasks, df, img_size, dataset_name):
#     model = None
#     if dataset_name == "botit": 
#         model = BotitCNNModel1(img_size, df)
#     elif dataset_name == "fashion":
#         model = CNNModel1(img_size=img_size, df=df)  # Initialize the model with the specified img_size

#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# def main():
#     parser = argparse.ArgumentParser(description="Test a multitask learning model.")
#     parser.add_argument('--data_dir', type=str, required=True, help="Directory for image data")
#     parser.add_argument('--dataset_name', type=str, default="botit", choices=["fashion", "botit", "xray"], help="the name of the used dataset for training")
#     parser.add_argument('--styles_csv_path', type=str, required=True, help="Path to styles CSV file")
#     parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
#     parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
#     parser.add_argument('--img_size', type=int, default=128, help="Size of the images")  # Add img_size parameter
#     args = parser.parse_args()

#     # Load the dataset
#     df = pd.read_csv(args.styles_csv_path, on_bad_lines='skip')
#     #df = df[:10000]

#     # Data augmentation and normalization
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(args.img_size),  # Use args.img_size here
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     # Create the main dataset
#     if args.dataset_name == "botit":
#         dataset = BotitDataset(df, args.data_dir, args.img_size, transform)
#     elif args.dataset_name == "fashion":
#         dataset = FashionDataset(df, args.data_dir, img_size=args.img_size, transform=transform)  # Pass img_size
#     elif args.dataset_name == "xray":
#         dataset = XrayDataset(df, args.data_dir, img_size=args.img_size, transform=transform)  # Pass img_size
#     else:
#         raise ValueError("Invalid dataset name. Please specify")

#     # Create DataLoader for test split
#     test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

#     # Load the model
#     model = load_model(args.model_path, num_tasks=len(df.columns) - 1, df=df, img_size=args.img_size, dataset_name=args.dataset_name)  # Pass img_size

#     # Initialize the loss function
#     criterion = nn.CrossEntropyLoss()

#     # Instantiate the tester and run the test
#     tester = MTLTester(model, test_loader, criterion)
#     tester.test()

# if __name__ == "__main__":
#     main()

# Define supported datasets
# dataset_registry = {
#     "fashion": lambda df, data_dir, img_size, transform: FashionDataset(df, data_dir, img_size, transform),
#     "botit": lambda df, data_dir, img_size, transform: BotitDataset(df, data_dir, img_size, transform),
#     "home_office": lambda df, data_dir, img_size, transform: OfficeHomeDataset(df, data_dir, img_size, transform),
#     "xray": lambda df, data_dir, img_size, transform: XrayDataset(df, data_dir, img_size, transform)
# }

# # Define supported models
# model_registry = {
#     "fashion": {
#         "CNN_model1": lambda img_size, df: CNNModel1(img_size, df),
#         "VGG16": lambda img_size, df: FashionVGG16(img_size, df),
#         "ResNet": lambda img_size, df: FashionResNet(img_size, df),
#         "efficientnet": lambda img_size, df: FashionEfficientNet(img_size, df),
#         "densenet": lambda img_size, df: FashionDenseNet(img_size, df),
#         "mobilenet": lambda img_size, df: FashionMobileNet(img_size, df),
#         "inception": lambda img_size, df: FashionInceptionNet(img_size, df),
#         "ViT": lambda img_size, df: FashionViT(img_size, df),
#         "swing": lambda img_size, df: FashionSwinTransformer(img_size, df)
#     },
#     "botit": {
#         "CNN_model1": lambda img_size, df: BotitCNNModel1(img_size, df),
#         "VGG16": lambda img_size, df: BotitVGG16(img_size, df),
#         "ResNet": lambda img_size, df: BotitResNet(img_size, df),
#         "efficientnet": lambda img_size, df: BotitEfficientNet(img_size, df),
#         "densenet": lambda img_size, df: BotitDenseNet(img_size, df),
#         "mobilenet": lambda img_size, df: BotitMobileNet(img_size, df),
#         "inception": lambda img_size, df: BotitInceptionNet(img_size, df),
#         "ViT": lambda img_size, df: BotitViT(img_size, df),
#         "swing": lambda img_size, df: BotitSwinTransformer(img_size, df)
#     },
#     "xray": {
#         "CNN_model1": lambda img_size, df: XrayCNNModel1(img_size, df),
#         "VGG16": lambda img_size, df: XrayVGG16(img_size, df),
#         "ResNet": lambda img_size, df: XrayResNet(img_size, df),
#         "efficientnet": lambda img_size, df: XrayEfficientNet(img_size, df)
#     }
# }

# def load_model(model_name, dataset_name, img_size, df, model_path):
#     if dataset_name not in model_registry:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")
#     if model_name not in model_registry[dataset_name]:
#         raise ValueError(f"Unsupported model: {model_name} for dataset {dataset_name}")

#     # Instantiate the model
#     model = model_registry[dataset_name][model_name](img_size, df)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# def main():
#     parser = argparse.ArgumentParser(description="Test a multitask learning model.")
#     parser.add_argument('--data_dir', type=str, required=True, help="Directory for image data")
#     parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_registry.keys(), help="The name of the dataset")
#     parser.add_argument('--model_name', type=str, required=True, help="The name of the model to use")
#     parser.add_argument('--styles_csv_path', type=str, required=True, help="Path to styles CSV file")
#     parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
#     parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoader")
#     parser.add_argument('--img_size', type=int, default=128, help="Size of the images")
#     args = parser.parse_args()

#     # Load the dataset CSV
#     df = pd.read_csv(args.styles_csv_path, on_bad_lines='skip')

#     # Data augmentation and normalization
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(args.img_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     # Create the dataset using the registry
#     if args.dataset_name not in dataset_registry:
#         raise ValueError(f"Unsupported dataset: {args.dataset_name}")
#     dataset = dataset_registry[args.dataset_name](df, args.data_dir, args.img_size, transform)

#     # Create DataLoader
#     test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

#     # Load the model
#     model = load_model(args.model_name, args.dataset_name, args.img_size, df, args.model_path)

#     # Initialize the loss function
#     criterion = nn.CrossEntropyLoss()

#     # Instantiate the tester and run the test
#     tester = MTLTester(model, test_loader, criterion)
#     tester.test()

# if __name__ == "__main__":
#     main()



# Dataset registry
dataset_registry = {
    "fashion": lambda df, data_dir, img_size, transform: FashionDataset(df, data_dir, img_size, transform),
    "botit": lambda df, data_dir, img_size, transform: BotitDataset(df, data_dir, img_size, transform),
    "home_office": lambda df, data_dir, img_size, transform: OfficeHomeDataset(df, data_dir, img_size, transform),
    "xray": lambda df, data_dir, img_size, transform: XrayDataset(df, data_dir, img_size, transform)
}

# Model registry
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
    "xray": {
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

def load_model(model_name, dataset_name, img_size, df, model_path):
    if dataset_name not in model_registry:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if model_name not in model_registry[dataset_name]:
        raise ValueError(f"Unsupported model: {model_name} for dataset {dataset_name}")

    # Instantiate the model
    model = model_registry[dataset_name][model_name](img_size, df)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Test a multitask learning model.")
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


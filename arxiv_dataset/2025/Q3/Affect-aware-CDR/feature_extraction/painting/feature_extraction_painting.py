#!/usr/bin/env python3
# coding: utf-8

"""
Extract 2048D features from painting images using a pre-trained ResNet50 model.
Saves the extracted features to mozart-crossmodal/data/paintings/painting_features_2048D.csv.

Authors:
- Bereket A. Yilma <name.surname@artaicare.com>
"""


import os
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet50
resnet50 = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(
    *list(resnet50.children())[:-1]
)  # Remove final FC layer
feature_extractor.to(device)
feature_extractor.eval()  # Set to evaluation mode

# Image transformation (match ImageNet preprocessing)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PAINTING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "paintings")
IMAGE_FOLDER = os.path.join(PAINTING_DATA_DIR, "images")
VA_CSV = os.path.join(PAINTING_DATA_DIR, "painting_valence_arousal.csv")
OUTPUT_CSV = os.path.join(PAINTING_DATA_DIR, "painting_features_2048D.csv")

# Load the painting valence-arousal CSV file
try:
    painting_va = pd.read_csv(VA_CSV)
except FileNotFoundError:
    raise FileNotFoundError(f"V-A CSV not found at {VA_CSV}")
except Exception as e:
    raise Exception(f"Error loading V-A CSV: {str(e)}")

# Store extracted features
features_list = []

# Process each image
for idx, row in tqdm(
    painting_va.iterrows(), total=len(painting_va), desc="Extracting Features"
):
    image_id = row["ID"]
    image_path = os.path.join(
        IMAGE_FOLDER, f"{image_id}.jpg"
    )  # Ensure image names match IDs

    # Load and preprocess the image
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Extract features (2048D vector)
        with torch.no_grad():
            features = feature_extractor(image)
            features = torch.flatten(features).cpu().numpy()  # Flatten to 1D

        # Append results
        features_list.append([image_id] + features.tolist())
    else:
        print(f"Warning: Image {image_id}.jpg not found!")

# Convert to DataFrame
feature_columns = [f"f{i}" for i in range(2048)]
features_df = pd.DataFrame(features_list, columns=["ID"] + feature_columns)

# Save to CSV
try:
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Painting feature extraction complete! Features saved to {OUTPUT_CSV}")
except Exception as e:
    raise Exception(f"Error saving output CSV: {str(e)}")

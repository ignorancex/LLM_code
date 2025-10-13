import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import cv2


class FashionDataset(Dataset):
    def __init__(self, dataframe, data_dir, img_size,transform=None):
        self.dataframe = dataframe.dropna(subset=['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage'])
        self.dataframe = self.dataframe[self.dataframe['id'].apply(lambda x: os.path.exists(os.path.join(data_dir, f"{x}.jpg")))] # Filter out rows where the image does not exist
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.label_encoders = {
            'gender': LabelEncoder(),
            'masterCategory': LabelEncoder(),
            'subCategory': LabelEncoder(),
            'articleType': LabelEncoder(),
            'baseColour': LabelEncoder(),
            'season': LabelEncoder(),
            'usage': LabelEncoder()
        }
        # Fit the label encoders on the available data
        for label in self.label_encoders:
            self.label_encoders[label].fit(self.dataframe[label].unique())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{row['id']}.jpg")
        
        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}, skipping this sample.")
            return None  # Return None or handle this case as needed

        img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
        img = cv2.resize(img, (self.img_size, self.img_size))  # Resize image
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        # Encode labels
        labels = [
            self.label_encoders['gender'].transform([row['gender']])[0],
            self.label_encoders['masterCategory'].transform([row['masterCategory']])[0],
            self.label_encoders['subCategory'].transform([row['subCategory']])[0],
            self.label_encoders['articleType'].transform([row['articleType']])[0],
            self.label_encoders['baseColour'].transform([row['baseColour']])[0],
            self.label_encoders['season'].transform([row['season']])[0],
            self.label_encoders['usage'].transform([row['usage']])[0]
        ]
        
        return img, torch.tensor(labels)

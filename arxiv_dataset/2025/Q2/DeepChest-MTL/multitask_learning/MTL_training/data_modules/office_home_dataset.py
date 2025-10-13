import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import cv2


class OfficeHomeDataset(Dataset):
    def __init__(self, dataframe, data_dir, img_size,transform=None):
        self.dataframe = dataframe.dropna(subset=[
            'path', 
            'domain', 
            'category'
        ])
        #self.dataframe = self.dataframe[self.dataframe['product_id'].apply(lambda x: os.path.exists(os.path.join(data_dir, f"{x}.jpg")))] # Filter out rows where the image does not exist
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.label_encoders = {
            'domain': LabelEncoder(),
            'category': LabelEncoder(),
        }
        # Fit the label encoders on the available data
        for label in self.label_encoders:
            self.label_encoders[label].fit(self.dataframe[label].unique())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}, skipping this sample.")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)  # Default placeholder image
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)  # Default placeholder image
            else:
                img = cv2.resize(img[..., ::-1], (self.img_size, self.img_size))  # Resize and convert to RGB
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        # Encode labels
        labels = [
            self.label_encoders['domain'].transform([row['domain']])[0],
            self.label_encoders['category'].transform([row['category']])[0],
        ]
        
        return img, torch.tensor(labels)

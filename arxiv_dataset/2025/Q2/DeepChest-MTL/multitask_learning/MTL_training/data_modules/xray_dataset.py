import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import cv2


class XrayDataset(Dataset):
    def __init__(self, dataframe, data_dir, img_size,transform=None):
        self.dataframe = dataframe.dropna(subset=[
            'Image', 
            'Atelectasis', 
            'Cardiomegaly', 
            'Consolidation', 
            'Edema', 
            'Effusion', 
            'Emphysema', 
            'Fibrosis', 
            'Hernia',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pleural_Thickening',
            'Pneumonia',
            'Pneumothorax'
        ])
        print("len before filtering ", self.dataframe.shape)
        self.dataframe = self.dataframe[self.dataframe['Image'].apply(lambda x: os.path.exists(os.path.join(data_dir, f"{x}")))] # Filter out rows where the image does not exist
        print("len after filtering ", self.dataframe.shape)
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.label_encoders = {
            'Atelectasis': LabelEncoder(),
            'Cardiomegaly': LabelEncoder(),
            'Consolidation': LabelEncoder(),
            'Edema': LabelEncoder(),
            'Effusion': LabelEncoder(),
            'Emphysema': LabelEncoder(),
            'Fibrosis': LabelEncoder(),
            'Hernia': LabelEncoder(),
            'Infiltration': LabelEncoder(),
            'Mass': LabelEncoder(),
            'Nodule': LabelEncoder(),
            'Pleural_Thickening': LabelEncoder(),
            'Pneumonia': LabelEncoder(),
            'Pneumothorax': LabelEncoder()
        }
        # Fit the label encoders on the available data
        for label in self.label_encoders:
            self.label_encoders[label].fit(self.dataframe[label].unique())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{row['Image']}")

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
            self.label_encoders['Atelectasis'].transform([row['Atelectasis']])[0],
            self.label_encoders['Cardiomegaly'].transform([row['Cardiomegaly']])[0],
            self.label_encoders['Consolidation'].transform([row['Consolidation']])[0],
            self.label_encoders['Edema'].transform([row['Edema']])[0],
            self.label_encoders['Effusion'].transform([row['Effusion']])[0],
            self.label_encoders['Emphysema'].transform([row['Emphysema']])[0],
            self.label_encoders['Fibrosis'].transform([row['Fibrosis']])[0],
            self.label_encoders['Hernia'].transform([row['Hernia']])[0],
            self.label_encoders['Infiltration'].transform([row['Infiltration']])[0],
            self.label_encoders['Mass'].transform([row['Mass']])[0],
            self.label_encoders['Nodule'].transform([row['Nodule']])[0],
            self.label_encoders['Pleural_Thickening'].transform([row['Pleural_Thickening']])[0],
            self.label_encoders['Pneumonia'].transform([row['Pneumonia']])[0],
            self.label_encoders['Pneumothorax'].transform([row['Pneumothorax']])[0],
        ]
        
        return img, torch.tensor(labels)

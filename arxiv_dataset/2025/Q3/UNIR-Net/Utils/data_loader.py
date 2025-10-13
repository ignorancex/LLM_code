import os
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class InferenceDataset(Dataset):
    def __init__(self, lowLightDir):
        self.lowLightDir  = lowLightDir 
        self.image_files = natsort.natsorted(os.listdir(lowLightDir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        lowLightImg = Image.open(os.path.join(self.lowLightDir, image_file))
        transform = ToTensor()
        lowLightImg = transform(lowLightImg)
        return lowLightImg  
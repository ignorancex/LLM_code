import glob
from pathlib import Path

from PIL import Image, ImageOps
from tqdm.auto import tqdm

import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data as data

import numpy as np

import pandas as pd

from typing import List
import pickle as pkl

import argparse

parser = argparse.ArgumentParser(description='Create feature vector database')
parser.add_argument('--model_name', type=str, default='resnet50', help='timm model name')
parser.add_argument('--img_size', type=int, default=224, help='Image size')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

class ThroatDataset(data.Dataset):
    def __init__(self, path_list: List, transform=None, exif_transpose: bool=True):
        super().__init__()
        
        self._transform = transform
        self._exif_transpose = exif_transpose
        self.samples = []

        # check CSV file
        if str(path_list[0]).endswith('.csv'):
            for csv_path in path_list:
                tmp_df = pd.read_csv(csv_path)
                self.samples.extend(tmp_df['path'].values)
        else:
            self.samples = path_list.copy()

    def __getitem__(self, index):
        image_path = self.samples[index]

        image = Image.open(image_path).convert('RGB')
        if (self._exif_transpose):
            image = ImageOps.exif_transpose(image)

        # Pytorch transform
        if self._transform:
            image = self._transform(image)

        return image, image_path

    def __len__(self):
        return len(self.samples)

if __name__=="__main__":
    args = parser.parse_args()

    # Load model
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model = model.cuda()
    model.eval()

    # Load dataset (14k images)
    root_dir = Path('/data/new_throat_screening_AI/data_path/')
    path_list = [
        root_dir/'train_good_throat.csv',
        root_dir/'test_good_throat.csv',
    ]
    
    data_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    throat_dataset = ThroatDataset(path_list, transform=data_transform)
    throat_loader = data.DataLoader(throat_dataset, batch_size=args.batch_size, shuffle=False)

    # Extract feature vector
    feature_vectors = []
    image_paths = []

    for batch_data in tqdm(throat_loader):
        img_batch, img_paths = batch_data
        image_paths.extend(img_paths)
        
        img_batch = img_batch.to('cuda')
        
        # extract query feature
        with torch.backends.cudnn.flags(deterministic=True, benchmark=True):
            with torch.no_grad():
                feature = model(img_batch)
        
        # normalize to unit norm
        feature = feature.detach().cpu()
        feature = F.normalize(feature, p=2, dim=1)
        feature_vectors.append(feature)

    feature_vectors = np.concatenate(feature_vectors)

    # Save feature vector
    feature_dict = {}
    for i, img_path in enumerate(image_paths):
        feature_dict[img_path] = feature_vectors[i]

    with open(f'{args.model_name}_feature_dict_v2.pkl', 'wb') as f:
        pkl.dump(feature_dict, f)
import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image1, image2, label = data['image1'], data['image2'], data['label']
        return {'image1': F.to_tensor(image1), 'image2': F.to_tensor(image2), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size1, size2):
        self.size1 = size1
        self.size2 = size2

    def __call__(self, data):
        image1, image2, label = data['image1'], data['image2'], data['label']

        return {'image1': F.resize(image1, self.size1), 'image2': F.resize(image2, self.size2), 'label': F.resize(label, self.size1, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image1, image2, label = data['image1'], data['image2'], data['label']

        if random.random() < self.p:
            return {'image1': F.hflip(image1), 'image2': F.hflip(image2), 'label': F.hflip(label)}

        return {'image1': image1, 'image2': image2, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image1, image2, label = data['image1'], data['image2'], data['label']

        if random.random() < self.p:
            return {'image1': F.vflip(image1), 'image2': F.vflip(image2), 'label': F.vflip(label)}

        return {'image1': image1, 'image2': image2, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']
        image1 = F.normalize(image1, self.mean, self.std)
        image2 = F.normalize(image2, self.mean, self.std)
        return {'image1': image1, 'image2': image2, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size1, size2, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size1, size1),(size2, size2)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size1, size1),(size2, size2)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image1 = self.rgb_loader(self.images[idx])
        image2 = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image1': image1, 'image2': image2, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size1, size2):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            Resize((size1, size1), (size2, size2)),
            ToTensor(),
            Normalize()
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image1 = self.rgb_loader(self.images[self.index])
        image2 = self.rgb_loader(self.images[self.index])
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.transform(image2).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image1, image2, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
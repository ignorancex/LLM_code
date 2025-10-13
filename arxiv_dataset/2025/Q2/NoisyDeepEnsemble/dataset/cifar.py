import os
import sys

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "cifar10")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()
        self.num_classes = 10
    
    def download(self):
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=True)
    
    def load_data(self):
        if self.train:
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=False)
        else:
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=False)
        
        images = []
        labels = []
        for image, label in dataset:
            images.append(image)
            labels.append(label)
        
        return images, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": label,
        }

class CIFAR100Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "cifar100")
        self.transform = transform
        self.train = train

        if download:
            self.download()
        self.num_classes = 100
        
        self.images, self.labels = self.load_data()
    
    def download(self):
        torchvision.datasets.CIFAR100(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR100(root=self.root, train=False, download=True)
    
    def load_data(self):
        if self.train:
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=False)
        else:
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=False, download=False)
        
        images = []
        labels = []
        for image, label in dataset:
            images.append(image)
            labels.append(label)
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": label,
        }

class CIFAR10CorruptDataset(Dataset):
    num_images_per_corruption = 10000
    c_list = ['gaussian_noise',
                'gaussian_blur',
                'spatter',
                'defocus_blur',
                'elastic_transform',
                'speckle_noise',
                'glass_blur',
                'motion_blur',
                'contrast',
                'saturate',
                'shot_noise',
                'snow',
                'pixelate',
                'fog',
                'impulse_noise',
                'frost',
                'jpeg_compression',
                'zoom_blur',
                'brightness']
    def __init__(self, root, severity: int, transform=None, download=False, train=False):
        self.root = os.path.join(root, "CIFAR-10-C", "CIFAR-10-C")
        self.transform = transform
        self.severity = severity
        self.train = train
        
        assert self.severity in range(1, 6)
        self.images, self.labels = self.load_data()
        self.num_classes = 10
    
    def load_data(self):
        # We assume first 10000 images are corrupted images with sevirity 1. 
        # Overall, we have 50000 images in CIFAR-10 dataset.
        labels_path = os.path.join(self.root, "labels.npy")
        labels = np.load(labels_path)
        assert labels.shape[0] == 50000, f"Labels shape is {labels.shape}, We expect 50000 labels"
        labels_sub = labels[:self.num_images_per_corruption]
        
        c_images = []
        c_labels = []
        for c_type in self.c_list:
            data_path = os.path.join(self.root, c_type + ".npy")
            assert os.path.exists(data_path), f"Data path {data_path} does not exist"
            data = np.load(data_path)
            assert data.shape[0] == 50000, f"Data shape is {data.shape}, We expect 50000 images"
            data = data[self.num_images_per_corruption * (self.severity - 1): self.num_images_per_corruption * self.severity]
            c_images.append(data)
            c_labels.append(labels_sub)

        c_images = np.stack(c_images, axis=0)  # shape: (20, 10000, 32, 32, 3)
        c_images = np.reshape(c_images, (19 * 10000, 32, 32, 3))  
        c_labels = np.stack(c_labels, axis=0)
        c_labels = np.reshape(c_labels, (19 * 10000))
        
        return c_images, c_labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return {
            "index": idx,
            "image": img,
            "label": label
        }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10 = CIFAR10Dataset(root="data", transform=transform, download=True, train=True)
    cifar100 = CIFAR100Dataset(root="data", transform=transform, download=True, train=True)

    print(len(cifar10))
    print(len(cifar100))
    print(cifar10[0])
    print(cifar100[0])
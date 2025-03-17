from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

class ImageNet_Format(Dataset):
    def __init__(self, path, transform=None):
        self.data = []
        self.transform = transform
        classes = np.loadtxt('imagenet-classes.txt', dtype='str')
        assert len(classes) == 1000
        for i in range(len(classes)):
            new_path = os.path.join(path, classes[i])
            if not os.path.isdir(new_path):
                continue
            for img in sorted(os.listdir(new_path)):
                self.data.append((os.path.join(new_path, img), i))
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class ImageNet_Subset(Dataset):
    def __init__(self, imagenet_path, subset_file, transform=None):
        self.data = []
        self.transform = transform
        classes = np.loadtxt('imagenet-classes.txt', dtype='str')
        assert len(classes) == 1000
        with open(subset_file) as f:
            for line in f:
                sub_path = line.rstrip()
                new_path = os.path.join(imagenet_path, sub_path)
                self.data.append(
                    (new_path, list(classes).index(sub_path.split("/")[0])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class ImageNetOOD(Dataset):
    def __init__(self, imagenet_path, subset_file, transform=None):
        self.data = []
        self.transform = transform
        with open(subset_file) as f:
            for line in f:
                sub_path = line.rstrip()
                new_path = os.path.join(imagenet_path, sub_path)
                self.data.append(
                    (new_path, -1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class Generic_Subset(Dataset):
    def __init__(self, path, subset_file, transform=None):
        self.data = []
        self.transform = transform
        with open(subset_file) as f:
            for line in f:
                self.data.append(os.path.join(path, line.rstrip()))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
        

class ImageNetOOD_standalone(Dataset):
    def __init__(self, path, transform=None):
        self.data = []
        self.transform = transform
        for img in sorted(os.listdir(path)):
            self.data.append(os.path.join(path, img))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
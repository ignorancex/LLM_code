import os
import sys

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNISTDataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "mnist")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()
    
    def download(self):
        torchvision.datasets.MNIST(root=self.root, train=True, download=True)
        torchvision.datasets.MNIST(root=self.root, train=False, download=True)
    
    def load_data(self):
        if self.train:
            dataset = torchvision.datasets.MNIST(root=self.root, train=True, download=False)
        else:
            dataset = torchvision.datasets.MNIST(root=self.root, train=False, download=False)
        
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
            "path": None
        }

if __name__ == "__main__":
    dataset = MNISTDataset(root="data", download=True)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]["label"])
    print(dataset[0]["path"])
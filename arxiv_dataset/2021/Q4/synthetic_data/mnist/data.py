import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image, ImageOps

#############################
#                           #
#           PATHS           #
#                           #
#############################

data_path = "./data" 
font_data_path = "./data/font"

#############################
#                           #
#         TRANSFORMS        #
#                           #
#############################

class Invert(object):
    def __call__(self, tens):
        tens = 1 - tens
        return tens

    def __repr__(self):
        return self.__class__.__name__

class Blur(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, img):
        arr = np.array(img)
        kwidth = 2 * int(np.random.rand() * self.alpha * 5) + 1
        kheight = 2 * int(np.random.rand() * self.alpha * 5) + 1
        out = cv2.blur(src=arr, ksize=(kwidth, kheight))
        return out

class Threshold(object):
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, tens):
        out = tens.clone()
        out[out > self.threshold] = 1
        return out

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

#############################
#                           #
#        FONT DATASET       #
#                           #
#############################

def font_train_transforms(alpha=1.0):
    return transforms.Compose([
        transforms.RandomAffine(15*alpha, translate=(.15*alpha, .15*alpha), 
            scale=(1-.25*alpha, 1+.05 * alpha), 
            shear=40*alpha,fillcolor=255),
        transforms.RandomPerspective(distortion_scale=0.5*alpha, p=alpha, 
            fill=255),
        Blur(alpha * .45),
        transforms.ToTensor(),
        Invert(),
        Threshold()
    ])

class FontDataset(datasets.vision.VisionDataset):
    def __init__(self, root=font_data_path, 
            transform=None, target_transform=None, use_alts=True):
        if transform is None:
            transform = font_train_transforms(0)

        super(FontDataset, self).__init__(root, transform=transform, 
                target_transform=target_transform)
        self.samples = self.load_samples(root, use_alts=use_alts)

    def load_samples(self, root, use_alts):
        images = os.listdir(root)
        samples = []
        for image in images:
            spl = image.split(".")
            if not use_alts:
                if len(spl) == 3 and spl[1] != "A":
                    continue
            target = int(spl[0])
            inp = pil_loader(os.path.join(root, image))
            samples.append((inp, target))
        samples.sort(key=lambda x: x[1])
        return samples

    def adjust_alpha(self, alpha):
        self.transform = font_train_transforms(alpha)

    def __getitem__(self, index):
        # index == label
        sample, label = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.samples)


#############################
#                           #
#        MNIST DATASET      #
#                           #
#############################

def test_transforms(_=None):
    return transforms.Compose([
        transforms.ToTensor()
    ])

class MNISTDataset(datasets.MNIST):
    def __init__(self, root=data_path, train=False,
            transform=None, target_transform=None):
        if transform is None:
            transform = test_transforms()

        super(MNISTDataset, self).__init__(root, train=train, download=True,
                transform=transform, target_transform=target_transform)

#############################
#                           #
#        CONSTRUCTORS       #
#                           #
#############################

def make_font_trainloader(shuffle=True):
    return torch.utils.data.DataLoader(
        FontDataset(use_alts=False),
        batch_size=10, shuffle=shuffle)

def make_mnist_testloader(batch_size=64, shuffle=True):
    return torch.utils.data.DataLoader(
        MNISTDataset(train=False),
        batch_size=batch_size, shuffle=shuffle)


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

picto_path = "data/Picto"
gtsrb_test_img_path = "data/GTSRB_Test/Images"
gtsrb_test_label_path = "data/GTSRB_Test/GT-final_test.csv"


#############################
#                           #
#         TRANSFORMS        #
#                           #
#############################

class Normalize(object):
    def __init__(self, cutoff=1):
        self.cutoff = cutoff

    def __call__(self, img):
        return ImageOps.equalize(ImageOps.autocontrast(img, cutoff=self.cutoff))

    def __repr__(self):
        return """PIL.ImageOps.equalize(
            PIL.ImageOps.autocontrast(..., cutoff={0}))""".format(self.cutoff)

class Blur(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, img):
        arr = np.array(img)
        kwidth = 2 * int(np.random.rand() * self.alpha * 5) + 1
        kheight = 2 * int(np.random.rand() * self.alpha * 5) + 1
        out = cv2.blur(src=arr, ksize=(kwidth, kheight))
        return out

class Clamp(object):
    def __init__(self, min=0.0, max=1.0):
        self.min = min
        self.max = max

    def __call__(self, tens):
        return torch.clamp(tens, self.min, self.max)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(min={0}, max={1})'.format(self.min, self.max)

class Exposure(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, tens):
        out = tens.clone()
        out *= (torch.rand(1) - 0.4) * self.alpha * .8 + 1.0 
        return out

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

normalize = transforms.Compose(
    [transforms.ToPILImage(), Normalize(), transforms.ToTensor()])


#############################
#                           #
#        PICTO DATASET      #
#                           #
#############################

# images are padded white and either 80x100 or 100x100
def picto_preprocess_transforms():
    return transforms.Compose([
        transforms.Pad(20, fill=(255, 255, 255)),
        transforms.CenterCrop(100),
        transforms.Resize(64)
    ])

def picto_train_transforms(alpha=1.0):
    return transforms.Compose([
        transforms.ColorJitter(brightness=.8*alpha, contrast=.8*alpha, 
            saturation=.8*alpha, hue=.05*alpha),
        Blur(alpha),
        transforms.ToTensor(),
        Exposure(alpha),
        Clamp()
    ])

def picto_train_preprocess_transforms(alpha=1.0):
    return transforms.Compose([
        transforms.RandomAffine(5*alpha, translate=(.15*alpha, .15*alpha), 
            scale=(1-.35*alpha, 1+.05 * alpha), 
            shear=5*alpha,fillcolor=(0, 0, 0, 0)),
        transforms.RandomPerspective(distortion_scale=0.5*alpha, p=1,
            fill=(0, 0, 0, 0))
    ])

def make_mask(a):
    mask = np.ones(a.shape[:2]) * 255

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for ii in [i, a.shape[0] - i - 1]:
                for jj in [j, a.shape[1] - j - 1]:
                    if i == 0 and j == 0:
                        mask[ii,jj] = 0 
                    elif mask[ii,jj] == 0:
                        continue
                    elif a[ii,jj].sum() > 1.5 * 255:
                        if ii > 0 and mask[ii-1,jj] == 0 \
                                or jj > 0 and mask[ii,jj-1] == 0 \
                                or ii < a.shape[0] - 1 and mask[ii+1,jj] == 0 \
                                or jj < a.shape[1] - 1 and mask[ii,jj+1] == 0:
                            mask[ii,jj] = 0 
    return mask


def sample_background(sample):
    bg = torch.randn(sample.shape) * 255
    return bg


class PictoTrainDataset(datasets.vision.VisionDataset):
    def __init__(self, root=picto_path, 
            preprocess_transform=None, 
            transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        super(PictoTrainDataset, self).__init__(root, transform=transform)
    
        self.samples = self.load_samples(root)
        self.preprocess_transform = preprocess_transform 

    def load_samples(self, root):
        preprocess = picto_preprocess_transforms()
        images = os.listdir(root)
        samples = []
        for image in images:
            inp = pil_loader(os.path.join(root, image))
            inp = preprocess(inp)

            # create mask
            a = np.array(inp)
            mask = make_mask(a)
            target = int(image.split('.')[0])

            new_size = list(a.shape)
            new_size[-1] = new_size[-1] + 1
            new_img = np.zeros(new_size)
            new_img[:,:,0:-1] = a
            new_img[:,:,-1] = mask
            inp = Image.fromarray(new_img.astype(np.uint8))
            samples.append((inp, target))
        samples.sort(key=lambda x: x[1])
        return samples

    def adjust_alpha(self, alpha):
        if self.transform is not None:
            self.transform = picto_train_transforms(alpha)
        if self.preprocess_transform is not None:
            self.preprocess_transform = \
                    picto_train_preprocess_transforms(alpha)

    def __getitem__(self, index):
        # index == label
        sample, label = self.samples[index]
        if self.preprocess_transform is not None:
            sample = self.preprocess_transform(sample)

        a = np.array(sample)
        mask = a[:,:,-1]
        sample = a[:,:,:-1]

        background = sample_background(sample)
        sample[mask == 0] = background[mask == 0]
        sample = Image.fromarray(sample.astype(np.uint8))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, mask, label

    def __len__(self):
        return len(self.samples)


#############################
#                           #
#       GTSRB DATASET       #
#                           #
#############################

def gtsrb_test_transforms():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        Normalize(),
        transforms.ToTensor()
    ])

class GTSRBTestDataset(datasets.vision.VisionDataset):
    def __init__(self, root=gtsrb_test_img_path, 
            label_csv=gtsrb_test_label_path,
            transform=None, target_transform=None,
            picto_dataset=None):
        if transform is None:
            transform = gtsrb_test_transforms()

        super(GTSRBTestDataset, self).__init__(root,
                transform=transform, target_transform=target_transform)
        self.samples = self.load_samples(root, label_csv)

    def load_samples(self, root, label_csv):
        labels = pd.read_csv(label_csv, sep=';')
        label_dict = dict(zip(list(labels.Filename), list(labels.ClassId)))

        images = os.listdir(root)
        samples = []
        for image in images:
            try:
                target = int(label_dict[image])
                samples.append((image, target))
            except:
                continue
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


#############################
#                           #
#        CONSTRUCTORS       #
#                           #
#############################

def make_picto_trainloader(batch_size=43, shuffle=True):
    return torch.utils.data.DataLoader(
        PictoTrainDataset(
            preprocess_transform=picto_train_preprocess_transforms(0),
            transform=picto_train_transforms(0)),
        batch_size=batch_size, shuffle=shuffle)

def make_gtsrb_testloader(batch_size=128, shuffle=False):
    return torch.utils.data.DataLoader(
        GTSRBTestDataset(),
        batch_size=batch_size, shuffle=shuffle)


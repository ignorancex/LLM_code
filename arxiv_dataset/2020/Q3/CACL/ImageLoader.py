import numpy as np
import os
import torch
import time
import glob
import math
from matplotlib import pyplot as plt
from PIL import Image
import random 
import nibabel as nib
from nibabel.testing import data_path

from torch.utils.data import Dataset
from torchvision import transforms, utils
#@from skimage.transform import resize
import csv
#from scipy.ndimage.interpolation import rotate
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import random
import imgaug.augmenters as iaa
import yaml


EXTENSIONS = ['.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

class Loader(Dataset):
    def __init__(self, dataset_root, input_transform=True):
        self.images_root = dataset_root
        self.images = []
        self.masks = []
        self.labels = []

        classes = glob.glob(os.path.join(self.images_root, '*'))

        for i in range(len(classes)):
            label = int(os.path.basename(classes[i]))
            all_masks = glob.glob(os.path.join(classes[i],'*_mask.png'))
            for j in range(len(all_masks)):
                image_root = all_masks[j].replace('_mask.png','.png')
                mask_root = all_masks[j]
                self.images.append(image_root)
                self.masks.append(mask_root)
                self.labels.append(label)

        self.image_transform = transforms.ToTensor()
        self.mask_transform = transforms.ToTensor()
        with open('configs/test.yaml', 'r') as f:
            temp = yaml.full_load(f.read())

        crop_size = int(temp['INPUT']['CROP_SIZE'])
        re_size = int(temp['INPUT']['SIZE'])

        self.image_aug = iaa.Sequential([
                                                iaa.CenterCropToFixedSize(height=crop_size, width=crop_size),
                                                iaa.Resize(re_size),
                                                # iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace='RGB'),
                                                # iaa.ChannelShuffle(0.35),
                                                # iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False, fill_mode="constant", cval=0),
                                                # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                                                #iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                                                # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                                                # iaa.GammaContrast((0.5, 2.0)),
                                                # iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                                                #iaa.Affine(rotate=(-180, 180)),
                                                # iaa.Affine(shear=(-16, 16)),
                                                #iaa.Fliplr(0.5),
                                                #iaa.GaussianBlur(sigma=(0, 1.0))
                                            ])

    def __getitem__(self, index):
        image_root = self.images[index]
        mask_root = self.masks[index]
        label = self.labels[index]
        img = load_image(image_root)#.resize((64,64))
        msk = load_image(mask_root)#.resize((64,64))

        img = self.image_transform(img)
        msk = self.mask_transform(msk)
        '''
        images = None
        for i in range(len(filename)):
            img = image_path(self.images_root, filename[i])
            image = load_image(img)
            if self.input_transform is not None:
                image = self.input_transform(image)

            image = image.unsqueeze(0)
            if i == 0:
                images = image
            else:
                images = torch.cat((images, image), axis=0)
        '''
        return img, msk, label

    def __len__(self):
        return len(self.images)



if __name__ == "__main__":
    validationset = Loader('/Data/proteins/code/VAE-GAN/VQVAE2_experiment/validationset')
    #sampler1 = dist.data_sampler(testingset, shuffle=True, distributed=args.distributed)
    loader_validation = DataLoader(
        validationset, batch_size=16, num_workers=0
    )
    for batch_idx, (img, msk, label) in enumerate(loader_validation):
        #data = data.to(device)
        print('aaa')
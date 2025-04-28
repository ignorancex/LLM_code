import os
import random
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(train_dir, test_dir, cr, ycrcb=False, name=False, seed=42):
    train_images = []
    test_images = []

    for root, _, fnames in sorted(os.walk(train_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                train_images.append(path)

    for root, _, fnames in sorted(os.walk(test_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                test_images.append(path)

    train_dataset = DATADataset(train_images, cr=cr, ycrcb=ycrcb, name=name)
    test_dataset = DATADataset(test_images, cr=cr, ycrcb=ycrcb, name=name)

    return train_dataset, test_dataset


class DATADataset(data.Dataset):
    def __init__(self, imgs, cr, ycrcb=False, name=False):
        self.imgs = imgs
        self.cr = cr
        torch.manual_seed(42)
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            ])

        self.ycrcb = ycrcb
        self.name = name

    def __getitem__(self, index):
        path = self.imgs[index]
        name = Path(path).stem

        img = cv2.imread(path)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_ycrcb = self.tfs(img_ycrcb.copy())
        img_y = img_ycrcb[0, :, :][None, :, :]

        if self.ycrcb:
            img_y = [img_y, img_ycrcb]
        if self.name:
            return img_y, name
        else:
            return img_y

    def __len__(self):
        return len(self.imgs)

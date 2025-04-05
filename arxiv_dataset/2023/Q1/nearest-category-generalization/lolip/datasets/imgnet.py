import os

import joblib
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder

def mini_imagenet(data_dir="./data/"):
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
    ])

    batch_size = 128

    dset = ImageFolder(os.path.join(data_dir, "mini-imagenet", "trn"), transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=8)
    trnX, trny = [], []
    for x, y in tqdm(loader):
        trnX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
        trny.append(y.numpy().astype(np.int32))
    trnX, trny = np.concatenate(trnX, axis=0), np.concatenate(trny)

    dset = ImageFolder(os.path.join(data_dir, "mini-imagenet", "val"), transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=8)
    tstX, tsty = [], []
    for x, y in tqdm(loader):
        tstX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
        tsty.append(y.numpy().astype(np.int32))
    tstX, tsty = np.concatenate(tstX, axis=0), np.concatenate(tsty)

    return trnX, trny, tstX, tsty

def imagenet_subsample_100(random_seed, data_dir="./data/"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    batch_size = 64
    class_to_idx = joblib.load(
            os.path.join(data_dir, "imagenet/imagenet_cls2idx.pkl"))

    res = joblib.load(
            os.path.join(data_dir, f"imagenet/imagenet_subsample_100_{random_seed}_trn.pkl"))
    dset = ImageListDataset(res['img_paths'], res['classes'], transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=8)
    trnX, trny = np.zeros((len(dset), 224, 224, 3), np.float32), []
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = x.numpy().astype(np.float32).transpose(0, 2, 3, 1)
        trnX[i*batch_size: i*batch_size+len(x)] = x
        y = np.asarray([class_to_idx[i] for i in y])
        trny.append(y.astype(np.int32))
    trny = np.concatenate(trny, axis=0)
    #trnX, trny = [], []
    #for x, y in tqdm(loader):
    #    trnX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
    #    y = np.asarray([class_to_idx[i] for i in y])
    #    trny.append(y.astype(np.int32))
    #trnX, trny = np.concatenate(trnX, axis=0), np.concatenate(trny, axis=0)

    res = joblib.load(
            os.path.join(data_dir, f"imagenet/imagenet_subsample_50_{random_seed}_tst.pkl"))
    dset = ImageListDataset(res['img_paths'], res['classes'], transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=8)
    tstX, tsty = np.zeros((len(dset), 224, 224, 3), np.float32), []
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = x.numpy().astype(np.float32).transpose(0, 2, 3, 1)
        tstX[i*batch_size: i*batch_size+len(x)] = x
        y = np.asarray([class_to_idx[i] for i in y])
        tsty.append(y.astype(np.int32))
    tsty = np.concatenate(tsty, axis=0)
    #tstX, tsty = [], []
    #for x, y in tqdm(loader):
    #    tstX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
    #    y = np.asarray([class_to_idx[i] for i in y])
    #    tsty.append(y.astype(np.int32))
    #tstX, tsty = np.concatenate(tstX, axis=0), np.concatenate(tsty)

    return trnX, trny, tstX, tsty


def imagenet_subsample_100_feature(random_seed, model_name="resnet50",
        device="cuda", data_dir="./data/"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if model_name == "resnet50":
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    elif model_name == "resnext101":
        pretrained_model = torchvision.models.resnext101_32x8d(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    elif model_name == "vgg19":
        pretrained_model = torchvision.models.vgg19(pretrained=True)
        feature_extractor = pretrained_model.classifier
    else:
        raise ValueError(f"[imagenet_subsample_100_feature] Not supported model_name: {model_name}")
    if torch.cuda.device_count() > 1:
        feature_extractor = nn.DataParallel(feature_extractor)
    feature_extractor.to(device)
    feature_extractor.eval()

    class_to_idx = joblib.load(
            os.path.join(data_dir, "imagenet/imagenet_cls2idx.pkl"))

    res = joblib.load(
            os.path.join(data_dir, f"imagenet/imagenet_subsample_100_{random_seed}_trn.pkl"))
    dset = ImageListDataset(res['img_paths'], res['classes'], transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=8)
    trnX, trny = [], []
    for x, y in tqdm(loader):
        x = feature_extractor(x.to(device))
        trnX.append(x.flatten(1).cpu().detach().numpy().astype(np.float32))
        y = np.asarray([class_to_idx[i] for i in y])
        trny.append(y.astype(np.int32))
    trnX, trny = np.concatenate(trnX, axis=0), np.concatenate(trny, axis=0)

    assert len(trnX) == 100000, trnX.shape
    assert len(np.unique(trny)) == 1000, trny.shape

    res = joblib.load(
            os.path.join(data_dir, f"imagenet/imagenet_subsample_50_{random_seed}_tst.pkl"))
    dset = ImageListDataset(res['img_paths'], res['classes'], transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=8)
    tstX, tsty = [], []
    for x, y in tqdm(loader):
        x = feature_extractor(x.to(device))
        tstX.append(x.flatten(1).cpu().detach().numpy().astype(np.float32))
        y = np.asarray([class_to_idx[i] for i in y])
        tsty.append(y.astype(np.int32))
    tstX, tsty = np.concatenate(tstX, axis=0), np.concatenate(tsty)

    assert len(tstX) == 50000, tstX.shape
    assert len(np.unique(tsty)) == 1000, tsty.shape

    return trnX, trny, tstX, tsty

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageListDataset(VisionDataset):
    """Face Landmarks dataset."""

    def __init__(self, image_paths, targets, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.image_paths[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.image_paths)


def get_oos_features(model_name="resnext101", data_dir="./data", device="cuda"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if model_name == "resnet50":
        pretrained_model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "resnext101":
        pretrained_model = torchvision.models.resnext101_32x8d(pretrained=True)
    else:
        raise ValueError(f"[imagenet_subsample_100_feature] Not supported model_name: {model_name}")
    model = torch.nn.Sequential(*list(pretrained_model.children())[:-1])

    dset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "oos_examples/", transform=transform))
    loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False, num_workers=8)

    model.to(device)
    model.eval()

    fets, ys = [], []
    for x, y in tqdm(loader):
        x = model(x.to(device)).flatten(1)
        fets.append(x.cpu().detach().numpy().astype(np.float32))
        ys.append(y)
    fets = np.concatenate(fets, 0)
    ys = np.concatenate(ys)
    return fets, ys

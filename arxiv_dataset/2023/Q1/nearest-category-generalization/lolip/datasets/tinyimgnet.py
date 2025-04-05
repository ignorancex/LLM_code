import os
from functools import partial, reduce

from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder

def leaveout_rand(trnX, trny, tstX, tsty, n_leaveouts, inter_var, random_state):
    leave_out_cls = random_state.choice(np.unique(trny), size=n_leaveouts, replace=False)

    trn_ind = reduce(lambda x, y: np.logical_or(x, y),
                        [trny == c for c in leave_out_cls])
    tst_ind = reduce(lambda x, y: np.logical_or(x, y),
                        [tsty == c for c in leave_out_cls])
    rest = (trnX[trn_ind], tstX[tst_ind], trny[trn_ind], tsty[tst_ind])
    inter_var['rest_ys'] = (trny[trn_ind], tsty[tst_ind])
    trnX, trny = trnX[np.logical_not(trn_ind)], trny[np.logical_not(trn_ind)]
    tstX, tsty = tstX[np.logical_not(tst_ind)], tsty[np.logical_not(tst_ind)]

    cls_map = {yi: i for i, yi in enumerate(np.unique(trny))}
    for i in range(len(trny)):
        trny[i] = cls_map[trny[i]]
    for i in range(len(tsty)):
        tsty[i] = cls_map[tsty[i]]

    inter_var['leave_out_cls'] = leave_out_cls
    inter_var['cls_map'] = cls_map

    return trnX, trny, tstX, tsty, rest

def get_tinyimgnet():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trn_loader, tst_loader = get_loader(transform=transform)

    trnX, trny = [], []
    for x, y in tqdm(trn_loader):
        trnX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
        trny.append(y.numpy().astype(np.int32))
    trnX, trny = np.concatenate(trnX, axis=0), np.concatenate(trny)

    assert len(trnX) == 100000, trnX.shape
    assert len(np.unique(trny)) == 200

    tstX, tsty = [], []
    for x, y in tqdm(tst_loader):
        tstX.append(x.numpy().astype(np.float32).transpose(0, 2, 3, 1))
        tsty.append(y.numpy().astype(np.int32))
    tstX, tsty = np.concatenate(tstX, axis=0), np.concatenate(tsty)

    assert len(tstX) == 10000, tstX.shape
    assert len(np.unique(tsty)) == 200

    return trnX, trny, tstX, tsty

def get_loader(transform=None):
    is_valid_file = lambda x: "ipynb_checkpoints" not in x and "txt" not in x
    trn_ds = ImageFolder("./data/tiny-imagenet-200/train/",
            transform=transform, is_valid_file=is_valid_file)
    tst_ds = ImageFolder("./data/tiny-imagenet-200/val/",
            transform=transform, is_valid_file=is_valid_file)

    trn_loader = torch.utils.data.DataLoader(trn_ds,
            batch_size=64, shuffle=False, num_workers=8)
    tst_loader = torch.utils.data.DataLoader(tst_ds,
            batch_size=64, shuffle=False, num_workers=8)
    return trn_loader, tst_loader

def tinyimgnet_resnet50(auto_var, inter_var, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    trn_loader, tst_loader = get_loader(transform=transform)

    pretrained_model = torchvision.models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    trnX = []
    trny = []
    for x, y in tqdm(trn_loader):
        x = feature_extractor(x.to(device)).flatten(1)
        trnX.append(x.cpu().detach().numpy().astype(np.float32))
        trny.append(y.numpy().astype(np.int32))
    trnX, trny = np.concatenate(trnX, axis=0), np.concatenate(trny)

    assert len(trnX) == 100000, trnX.shape
    assert len(np.unique(trny)) == 200

    tstX = []
    tsty = []
    for x, y in tqdm(tst_loader):
        x = feature_extractor(x.to(device)).flatten(1)
        tstX.append(x.cpu().detach().numpy().astype(np.float32))
        tsty.append(y.numpy().astype(np.int32))
    tstX, tsty = np.concatenate(tstX, axis=0), np.concatenate(tsty)

    assert len(tstX) == 10000, tstX.shape
    assert len(np.unique(tsty)) == 200

    return trnX, trny, tstX, tsty

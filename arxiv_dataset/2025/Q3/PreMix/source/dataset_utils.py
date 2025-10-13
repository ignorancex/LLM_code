import torch

from PIL import Image
from torch.nn.utils.rnn import pad_sequence

def collate_features(batch, label_type: str = "int"):
    idx, slide_id, features, labels = zip(*batch)

    idx = torch.LongTensor(idx)
    seq_length = [len(seq) for seq in features]
    
    # implement pad mini-batch wise which pads the current mini-batch using
    # the biggest sequence length from that mini-batch
    features = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)

    mask = torch.arange(features.size(1))[None, :] < torch.tensor(seq_length)[:, None]
    mask = ~mask # (False: processed, True: padding)
    mask = mask[:, :features.shape[1]] # slice the masks tensor to have the same number of columns as the feature tensors

    if label_type == "float":
        labels = torch.FloatTensor([item[3] for item in batch])
    elif label_type == "int":
        labels = torch.LongTensor([item[3] for item in batch])
    
    return [idx, slide_id, features, mask, labels]

def pretrain_collate_features(batch):

    slide_id, features1, features2 = zip(*batch)
    seq_lengths1 = [len(seq) for seq in features1]
    seq_lengths2 = [len(seq) for seq in features2]

    features1 = pad_sequence(features1, batch_first=True, padding_value=0)
    features2 = pad_sequence(features2, batch_first=True, padding_value=0)

    mask1 = torch.arange(features1.size(1))[None, :] < torch.tensor(seq_lengths1)[:, None]
    mask2 = torch.arange(features2.size(1))[None, :] < torch.tensor(seq_lengths2)[:, None]
    mask1 = ~mask1 # (False: processed, True: padding)
    mask2 = ~mask2 # (False: processed, True: padding)
    mask1 = mask1[:, :features1.shape[1]] # slice the masks tensor to have the same number of columns as the feature tensors
    mask2 = mask2[:, :features2.shape[1]]

    return [slide_id, features1, features2, mask1, mask2]

def RandomZeroing(features, p=0.5):
    mask = torch.rand_like(features) > p
    return features * mask

def GaussianNoise(features, std=0.1):
    return features + torch.rand_like(features) * std

def RandomScaling(features, min_scale=0.9, max_scale=1.1):
    scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    return (features * scale)

def RandomCrop(features, crop_size=0.3):
    num_features = features.size(0)
    num_cropped_features = int(num_features * crop_size)
    cropped_indices = torch.randperm(num_features)[:num_cropped_features]
    features[cropped_indices, :] = 0
    return features

def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)

'''
From: A method for normalizing histology slides for quantitative analysis, 
M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun, 
C Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250

https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

'''

import copy
import numpy as np

def stain_norm(img, Io = 240, alpha = 1, beta = 0.15):

    original_img = copy.deepcopy(img)
                               
    HERef = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]

    # compute eigenvectors
    try:
        _ , eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    except:
        return original_img

    #eigvecs *= -1

    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues          
    That = ODhat.dot(eigvecs[:,1:3])

    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)     

    return Inorm
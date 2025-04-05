import os
import gc

from tqdm import tqdm
import numpy as np
import faiss
import joblib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_faiss_index(d, p):
    if p == 2:
        index = faiss.IndexFlatL2(d)
    elif p == np.inf:
        index = faiss.IndexFlat(d, faiss.METRIC_Linf)
    elif p == 1:
        index = faiss.IndexFlat(d, faiss.METRIC_L1)
    else:
        raise ValueError(f"[_get_nearest_oppo_dist] not supported norm: {p}")
    return index

def get_nearest_oppo_dist(X, y, norm, cache_filename=None):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    if cache_filename is not None and os.path.exists(cache_filename):
        print(f"[nearest_oppo_dist] Using cache: {cache_filename}")
        ret = joblib.load(cache_filename)
    else:
        print(f"[nearest_oppo_dist] cache {cache_filename} don't exist. Calculating...")

        ret = np.inf * np.ones(len(X))
        X = X.astype(np.float32, copy=False)

        for yi in tqdm(np.unique(y), desc="[nearest_oppo_dist]"):
            index = get_faiss_index(X.shape[1], p)
            index.add(X[y!=yi])

            idx = np.where(y==yi)[0]
            D, _ = index.search(X[idx], 1)
            if p == 2:
                D = np.sqrt(D)
            ret[idx] = np.minimum(ret[idx], D[:, 0])

            del index
            gc.collect()

        if cache_filename is not None:
            joblib.dump(ret, cache_filename)

    return ret

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_aug_nearest_oppo_dist(X, y, norm, n_samples=200, aug_rounds=5):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    X = X.transpose(0, 3, 1, 2)

    torch.manual_seed(0)
    dset = CustomTensorDataset((torch.from_numpy(X).float(), torch.from_numpy(y).long()),
                               transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=256, shuffle=False)
    augX, augy = [X], [y]
    for _ in range(aug_rounds):
        for xi, yi in loader:
            augX.append(xi.numpy())
            augy.append(yi.numpy())
    augX, augy = np.concatenate(augX, axis=0), np.concatenate(augy)
    augX = augX.reshape(len(augX), -1)

    random_state = np.random.RandomState(0)
    sample_ind = random_state.choice(len(X), size=n_samples)
    samX = X.reshape(len(X), -1)[sample_ind]
    samy = y[sample_ind]

    X = X.reshape(len(X), -1)
    X = X.astype(np.float32, copy=False)
    samX = samX.astype(np.float32, copy=False)

    ret = np.inf * np.ones(len(samX))
    for yi in tqdm(np.unique(y), desc="[nearest_oppo_dist]"):
        index = get_faiss_index(augX.shape[1], norm)
        index.add(augX[augy!=yi])

        idx = np.where(samy==yi)[0]
        D, _ = index.search(samX[idx], 1)
        if norm == 2:
            D = np.sqrt(D)
        ret[idx] = np.minimum(ret[idx], D[:, 0])

        del index
        gc.collect()

    ori_ret = np.inf * np.ones(len(samX))
    for yi in tqdm(np.unique(y), desc="[nearest_oppo_dist]"):
        index = get_faiss_index(X.shape[1], norm)
        index.add(X[y!=yi])

        idx = np.where(samy==yi)[0]
        D, _ = index.search(samX[idx], 1)
        if norm == 2:
            D = np.sqrt(D)
        ori_ret[idx] = np.minimum(ori_ret[idx], D[:, 0])

        del index
        gc.collect()

    return ori_ret, ret

#        #if X.size > 3*(10**9):
#        if None: # LSH
#            for yi in tqdm(np.unique(y), desc="[nearest_oppo_dist]"):
#                index = faiss.IndexLSH(X.shape[1], 2048)
#                index.train(X[y!=yi])
#                index.add (X[y!=yi])
#
#                #quantizer = get_index(X.shape[1], p)
#                #index = faiss.IndexIVFPQ(quantizer, X.shape[1], 16, 8, 8)
#
#                #index.train(X[y!=yi])
#                #index.add(X[y!=yi])
#                #index.nprobe = 10
#
#                idx = np.where(y==yi)[0]
#                _, I = index.search(X[idx], 1)
#                D = np.linalg.norm(X[y!=yi][I[:, 0]] - X[idx], ord=p, axis=1)
#
#                ret[idx] = np.minimum(ret[idx], D)
#
#                del index
#                gc.collect()
#        else:
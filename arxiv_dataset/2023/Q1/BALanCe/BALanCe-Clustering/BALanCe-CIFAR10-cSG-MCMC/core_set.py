import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb
import models
import torch.nn.functional as F
import glob

def acquire(model_dir, pool, pool_train, device, B=10):
    net = models.ResNet18().to(device)
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    dataset_train = MyDataset(pool_train)
    dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )

    origin_pos_lt = []
    pool_feature_lt = []
    labeled_set_feature_lt = []

    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pool_feature_lt = []
        tmp_origin_pos_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)
                outputs, feature = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_pool_feature_lt += list(feature.data.cpu().numpy())
                tmp_origin_pos_lt += list(pos.data.cpu().numpy())
        
        pool_feature_lt.append(tmp_pool_feature_lt)
        origin_pos_lt.append(tmp_origin_pos_lt)

        tmp_labeled_set_feature_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader_train):
                x, label = x.to(device), label.to(device)
                outputs, feature = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_labeled_set_feature_lt += list(feature.data.cpu().numpy())

        labeled_set_feature_lt.append(tmp_labeled_set_feature_lt)

    pool_feature_lt = np.array(pool_feature_lt)
    pool_feature_lt = np.transpose(pool_feature_lt, (1, 0, 2))
    pool_feature_lt = np.mean(pool_feature_lt, axis=1)

    labeled_set_feature_lt = np.array(labeled_set_feature_lt)
    labeled_set_feature_lt = np.transpose(labeled_set_feature_lt, (1, 0, 2))
    labeled_set_feature_lt = np.mean(labeled_set_feature_lt, axis=1)

    origin_pos_lt = origin_pos_lt[0]

    def furthest_first(X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    #chosen = init_centers(embedding_lt, B) 
    chosen = furthest_first(pool_feature_lt, labeled_set_feature_lt, B)
    sel_pos_lt = []
    for i in chosen:
        sel_pos_lt.append(origin_pos_lt[i])

    return sel_pos_lt


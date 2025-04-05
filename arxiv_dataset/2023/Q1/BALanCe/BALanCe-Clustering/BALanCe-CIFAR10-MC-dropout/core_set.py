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

def acquire(model, pool, pool_train, device, B=10, sample_num=20):
    model.eval()
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

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.classifier[3].register_forward_hook(get_activation('fc1'))

    origin_pos_lt = []
    acq_sample = []
    embedding_lt = []

    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            batch_size = x.size(0)
            pred = model(x, sample_num)
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            feature = activation['fc1']
            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()

            embedding_lt += list(feature)
            origin_pos_lt += list(pos.data.cpu().numpy())

    embedding_lt = np.array(embedding_lt)
    
    labeled_set_embedding_lt = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            batch_size = x.size(0)
            pred = model(x, sample_num)
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            feature = activation['fc1']
            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()

            labeled_set_embedding_lt += list(feature)

    labeled_set_embedding_lt = np.array(labeled_set_embedding_lt)
    
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
    chosen = furthest_first(embedding_lt, labeled_set_embedding_lt, B)
    sel_pos_lt = []
    for i in chosen:
        sel_pos_lt.append(origin_pos_lt[i])

    return sel_pos_lt


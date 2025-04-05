import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical

import os
from sklearn.metrics import pairwise_distances
from scipy import stats
from copy import deepcopy

def delta_coreset(model, pool, pool_train, device, B=3, M=10000, sample_num=20, sampling_index=4, class_num=10, score_batch_size=40):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    dataset_train = MyDataset(pool_train)
    dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    pool_pred_lt = []
    embedding_lt = []

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            #print(input)
            activation[name] = input[0].detach()
        return hook

    model.fc2.register_forward_hook(get_activation('fc2'))

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            batch_size = x.size(0)
            feature = activation['fc2']
            #print(f'feature size: {feature.size()}')

            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()

            pool_pred_lt += list(pred_exp)
            embedding_lt += list(feature)

    labeled_set_embedding_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            batch_size = x.size(0)
            feature = activation['fc2']
            #print(f'feature size: {feature.size()}')

            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()
            #pool_pred_lt += list(pred_exp)
            labeled_set_embedding_lt += list(feature)

    embedding_lt = np.array(embedding_lt)
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

    embedding_lt = np.array(embedding_lt)
    chosen = furthest_first(embedding_lt, labeled_set_embedding_lt, B)
    
    return None, chosen



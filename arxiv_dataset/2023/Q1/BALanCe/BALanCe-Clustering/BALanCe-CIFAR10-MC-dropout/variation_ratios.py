import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils

def variation_ratios(logits_b_K_C):
    '''
    copy from https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0/src
    '''
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) - torch.exp(
        torch.max(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0]
    )

def acquire(model, pool, device, B=10, sample_num=20):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    acq_sample = []
    origin_pos_lt = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            batch_acq_sample = variation_ratios(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)
            origin_pos_lt += list(pos.data.cpu().numpy())
 
    acq_sample = np.array(acq_sample)

    cluster2pos = pool.cluster2pos
    pos2cluster = pool.pos2cluster
    cluster_num_lt = [(cluster_label, len(cluster2pos[cluster_label])) for cluster_label in cluster2pos]
    cluster_num_lt.sort(key=lambda x:x[1])

    acquire_num_lt = []
    total_remain = 0

    for i in range(len(cluster_num_lt)):
        acquire_num_lt.append([cluster_num_lt[i][0], 0])
        total_remain += cluster_num_lt[i][1]

    assert total_remain >= B

    b = 0
    i = 0
    while b < B:
        while acquire_num_lt[i][1] >= cluster_num_lt[i][1]:
            i += 1
            i = i % len(acquire_num_lt)

        acquire_num_lt[i][1] += 1
        i += 1
        i = i % len(acquire_num_lt)
        b += 1

    cluster2subset_size = {}
    for i in range(len(acquire_num_lt)):
        cluster_label = acquire_num_lt[i][0]
        subset_size = acquire_num_lt[i][1]
        cluster2subset_size[cluster_label] = subset_size

    pos2utility = {origin_pos_lt[i]:acq_sample[i] for i in range(len(acq_sample))}
    cluster2pos_utility = {}
    sel_pos_lt = []
    for cluster_label in cluster2pos:
        cluster2pos_utility[cluster_label] = []
        for pos in cluster2pos[cluster_label]:
            cluster2pos_utility[cluster_label].append((pos, pos2utility[pos]))

        cluster2pos_utility[cluster_label].sort(key=lambda x:x[1], reverse=True)
        subset_size = cluster2subset_size[cluster_label]

        for i in range(subset_size):
            sel_pos_lt.append(cluster2pos_utility[cluster_label][i][0])

    return sel_pos_lt
 

    '''
    tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
    tmp_lt.sort(key=lambda x:x[1], reverse=True)
    pos_lt = []
    for i in range(B):
        pos = tmp_lt[i][0]
        pos_lt.append(pos)
    
    sel_pos_lt = [origin_pos_lt[pos] for pos in pos_lt]
    return acq_sample, sel_pos_lt
    '''


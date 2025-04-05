import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils
import random

def mean_stddev_acquisition_function(logits_b_K_C):
    '''
    copy from https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0/src
    '''
    return torch_utils.mean_stddev(logits_b_K_C)

def margin_score(logits_b_K_C):
    logits_b_K_C_exp = logits_b_K_C.exp_()
    logits_b_C_exp = torch.mean(logits_b_K_C_exp, dim=1)
    #print(f'logits_b_C_exp: {logits_b_C_exp.size()}')
    top2, _ = torch.topk(logits_b_C_exp, k=2, dim=1)
    #print(f'top2: {top2.size()}')
    margin_score = top2[:,0] - top2[:,1]
    return margin_score

def acquire(model, pool, pool_val, device, B=10, sample_num=20):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    origin_pos_lt = []
    acq_sample = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            #batch_acq_sample = mean_stddev_acquisition_function(pred)
            batch_acq_sample = margin_score(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)
            origin_pos_lt += list(pos.data.cpu().numpy())

    acq_sample = np.array(acq_sample)
   
    val_origin_pos_lt = []
    val_pred_lt = []
    val_label_lt = []

    dataset = MyDataset(pool_val)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )

    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            val_origin_pos_lt += list(pos.data.cpu().numpy())
            label_soft_pred = torch.mean(pred.exp_(), dim=1)
            val_pred_lt += list(label_soft_pred.data.cpu().numpy())
            label = label.view(-1)
            val_label_lt += list(label.data.cpu().numpy())

    val_pred_lt = np.array(val_pred_lt, dtype=np.float32)
    val_origin_pos_lt = np.array(val_origin_pos_lt)
    val_label_lt = np.array(val_label_lt)

    val_pos2index = {}
    for index in range(len(val_label_lt)):
        pos = val_origin_pos_lt[index]
        val_pos2index[pos] = index

    val_cluster2index = {}
    for cluster_label in pool_val.cluster2pos:
        val_cluster2index[cluster_label] = []
        for pos in pool_val.cluster2pos[cluster_label]:
            val_cluster2index[cluster_label].append(val_pos2index[pos])

    val_cluster2acc = {}
    for cluster_label in val_cluster2index:
        tmp_index = val_cluster2index[cluster_label]
        tmp_pred_lt = val_pred_lt[tmp_index]
        tmp_label_lt = val_label_lt[tmp_index]
        tmp_pred_lt = np.argmax(tmp_pred_lt, axis=1)
        acc = np.mean(tmp_pred_lt == tmp_label_lt)
        val_cluster2acc[cluster_label] = acc

    cluster2pos = pool.cluster2pos
    pos2cluster = pool.pos2cluster

    cluster_label_lt = []
    cluster_weight_lt = []

    for cluster_label in cluster2pos:
        cluster_label_lt.append(cluster_label)
        if cluster_label in val_cluster2acc:
            cluster_weight_lt.append((1.0-val_cluster2acc[cluster_label])*len(cluster2pos[cluster_label]))
        else:
            cluster_weight_lt.append(len(cluster2pos[cluster_label]))

    cluster_weight_lt = np.array(cluster_weight_lt)
    cluster_weight_lt = cluster_weight_lt/np.sum(cluster_weight_lt)
    cluster_label_lt = np.array(cluster_label_lt)

    perm_index_lt = [i for i in range(len(cluster_label_lt))]
    random.shuffle(perm_index_lt)
    perm_index_lt = np.array(perm_index_lt)
    cluster_label_lt = cluster_label_lt[perm_index_lt]
    cluster_weight_lt = cluster_weight_lt[perm_index_lt]

    cluster_subset_size_lt = []
    total_count = 0

    for i in range(len(cluster_label_lt)):
        cluster_label = cluster_label_lt[i]
        subset_size = int(B*cluster_weight_lt[i])
        cluster_subset_size_lt.append(subset_size)
        total_count += subset_size

    if total_count < B:
        i = 0
        left = B - total_count
        while left > 0:
            cluster_label = cluster_label_lt[i]
            if cluster_subset_size_lt[i] < len(cluster2pos[cluster_label]):
                cluster_subset_size_lt[i] += 1
                left -= 1
            i += 1
            i = i % len(cluster_label_lt)

    cluster2subset_size = {}
    for i in range(len(cluster_label_lt)):
        cluster_label = cluster_label_lt[i]
        cluster2subset_size[cluster_label] = cluster_subset_size_lt[i]

    pos2utility = {origin_pos_lt[i]:acq_sample[i] for i in range(len(acq_sample))}
    cluster2pos_utility = {}
    sel_pos_lt = []

    for cluster_label in cluster2pos:
        cluster2pos_utility[cluster_label] = []
        for pos in cluster2pos[cluster_label]:
            cluster2pos_utility[cluster_label].append((pos, pos2utility[pos]))
        cluster2pos_utility[cluster_label].sort(key=lambda x:x[1], reverse=False)
        subset_size = cluster2subset_size[cluster_label]

        for i in range(subset_size):
            sel_pos_lt.append(cluster2pos_utility[cluster_label][i][0])

    assert len(sel_pos_lt) == B
    return sel_pos_lt


    '''
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
    
    assert len(sel_pos_lt) == B
    return sel_pos_lt

    '''


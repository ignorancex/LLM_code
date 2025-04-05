import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical

import os

def choose_subset(pool_pred_lt, origin_pos_lt, subset_size, device, M=10000, sample_num=20, sampling_index=3, score_batch_size=4):
    if len(origin_pos_lt) <= subset_size:
        return origin_pos_lt
        
    #sub_pos_lt = []
    # K*1 at first
    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    pos_lt = []
    pool_size = len(origin_pos_lt)

    acq_sample = []
    start = 0
    while start < pool_size:
        #print(f'small b: {b}, start: {start}')
        end = start + score_batch_size
        if end > pool_size:
            end = pool_size
    
        # batch*K*C
        batch_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[start:end]).to(device)
        batch_pool_pred_lt_cur_cuda = torch.matmul(pool_pred_lt_pre_cuda[None,:,:,None], batch_pool_pred_lt_cuda[:,:,None,:])
        tmp_size = batch_pool_pred_lt_cur_cuda.size()
        # batch*K*C^b
        batch_pool_pred_lt_cur_cuda = batch_pool_pred_lt_cur_cuda.view((tmp_size[0], tmp_size[1], -1))
        # batch*C^b

        batch_p_y_cuda = torch.mean(batch_pool_pred_lt_cur_cuda, dim=1)
        cate_prob = Categorical(batch_p_y_cuda)
        # batch
        batch_entropy_p_y_cuda = cate_prob.entropy()
    
        # batch*K
        cate_prob = Categorical(batch_pool_pred_lt_cur_cuda)
        batch_entropy_p_w_y_cuda = cate_prob.entropy()
        # batch
        batch_mean_entropy_p_w_y_cuda = torch.mean(batch_entropy_p_w_y_cuda, dim=1)
        # batch
        batch_acq_sample_cuda = batch_entropy_p_y_cuda - batch_mean_entropy_p_w_y_cuda
    
        acq_sample += list(batch_acq_sample_cuda.cpu().numpy())

        start += score_batch_size
    

    tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
    tmp_lt.sort(key=lambda x:x[1], reverse=True)
    for i in range(subset_size):
        pos = tmp_lt[i][0]
        pos_lt.append(pos)
   
    sel_pos_lt = [origin_pos_lt[pos] for pos in pos_lt]
    
    return sel_pos_lt



def acquire(model, pool, pool_val, device, B=30, M=10000, sample_num=20, sampling_index=4, score_batch_size=40):
    model.eval()
    
    origin_pos_lt = []
    pool_pred_lt = []
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            origin_pos_lt += list(pos.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)
    pool_size = pool.size()

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
        if subset_size > len(cluster2pos[cluster_label]):
            subset_size = len(cluster2pos[cluster_label])
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

    print(f'cluster_weight_lt')
    print(cluster_weight_lt)
    print(f'cluster_subset_size_lt')
    print(cluster_subset_size_lt)

    cluster2subset_size = {}
    for i in range(len(cluster_label_lt)):
        cluster_label = cluster_label_lt[i]
        cluster2subset_size[cluster_label] = cluster_subset_size_lt[i]

    origin_pos2curr_pos = {pos:i for i, pos in enumerate(origin_pos_lt)}

    choose_pos_lt = []
    for i in range(len(cluster_label_lt)):
        cluster_label = cluster_label_lt[i]
        subset_size = cluster2subset_size[cluster_label]
        pos_lt = cluster2pos[cluster_label]
        index_lt = [origin_pos2curr_pos[pos] for pos in pos_lt]
        pred_lt = pool_pred_lt[index_lt]
        sub_pos_lt = choose_subset(pred_lt, pos_lt, subset_size, device=device, M=M, sample_num=sample_num, sampling_index=sampling_index, score_batch_size=score_batch_size)
        choose_pos_lt += sub_pos_lt

    assert len(choose_pos_lt) == B

    return choose_pos_lt
 
    
    '''
    cluster_num_lt = [(cluster_label, len(cluster2pos[cluster_label])) for cluster_label in cluster2pos]
    cluster_num_lt.sort(key=lambda x:x[1])
    
    #print(f'cluster_num_lt')
    #print(cluster_num_lt)
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

    #print(f'acquire_num_lt')
    #print(acquire_num_lt)

    origin_pos2curr_pos = {pos:i for i, pos in enumerate(origin_pos_lt)}

    choose_pos_lt = []
    for i in range(len(acquire_num_lt)):
        cluster_label = acquire_num_lt[i][0]
        subset_size = acquire_num_lt[i][1]
        pos_lt = cluster2pos[cluster_label]
        index_lt = [origin_pos2curr_pos[pos] for pos in pos_lt]
        pred_lt = pool_pred_lt[index_lt]
        sub_pos_lt = choose_subset(pred_lt, pos_lt, subset_size, device=device, M=M, sample_num=sample_num, sampling_index=sampling_index, score_batch_size=score_batch_size)
        choose_pos_lt += sub_pos_lt

    assert len(choose_pos_lt) == B

    return choose_pos_lt
    '''


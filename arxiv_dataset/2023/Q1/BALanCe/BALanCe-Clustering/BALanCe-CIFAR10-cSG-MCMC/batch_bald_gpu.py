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

    if subset_size <= sampling_index:
        for b in range(subset_size):
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
            for i in range(len(tmp_lt)):
                pos = tmp_lt[i][0]
                if pos not in pos_lt:
                    break
            pos_lt.append(pos)

            if b == sampling_index-1:
                break
        
            pos_pred_lt_cuda = torch.from_numpy(pool_pred_lt[pos]).to(device)
            pool_pred_lt_pre_cuda = torch.matmul(pool_pred_lt_pre_cuda[:,:,None], pos_pred_lt_cuda[:,None,:])
            # K*C^b
            pool_pred_lt_pre_cuda = pool_pred_lt_pre_cuda.view((sample_num, -1))
        
        
        sel_pos_lt = [origin_pos_lt[pos] for pos in pos_lt]
        return sel_pos_lt

    for b in range(sampling_index):
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
        for i in range(len(tmp_lt)):
            pos = tmp_lt[i][0]
            if pos not in pos_lt:
                break
        pos_lt.append(pos)

        if b == sampling_index-1:
            break
        
        pos_pred_lt_cuda = torch.from_numpy(pool_pred_lt[pos]).to(device)
        pool_pred_lt_pre_cuda = torch.matmul(pool_pred_lt_pre_cuda[:,:,None], pos_pred_lt_cuda[:,None,:])
        # K*C^b
        pool_pred_lt_pre_cuda = pool_pred_lt_pre_cuda.view((sample_num, -1))
        
    # K
    sum_cond_entropy_cuda = torch.zeros((sample_num,), device=device)
    for pos in pos_lt:
        pos_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[pos]).to(device)
        cate_prob = Categorical(pos_pool_pred_lt_cuda)
        cur_entropy_cuda = cate_prob.entropy()
        sum_cond_entropy_cuda += cur_entropy_cuda
    
    # K*M
    P_n_1_cuda = torch.ones((sample_num, M), device=device)

    for pos in pos_lt:
        sample_prob = np.mean(pool_pred_lt[pos], axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        sample_prob_cuda = torch.from_numpy(sample_prob)
        yn_lt = torch.multinomial(sample_prob_cuda, M, replacement=True)
        
        pos_pool_pred_lt = pool_pred_lt[pos]
        pos_pool_pred_lt_cuda = torch.from_numpy(pos_pool_pred_lt).to(device)
        P_n_1_cuda = P_n_1_cuda * pos_pool_pred_lt_cuda[:,yn_lt]
   
    for b in range(sampling_index, subset_size):
        start = 0
        acq_sample = []
        while start < pool_size:
            #print(f'large b: {b}, start: {start}')
            end = start + score_batch_size
            if end > pool_size:
                end = pool_size
            # batch*k*class
            batch_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[start:end]).to(device)
            # batch*class*M
            batch_P_n_1_P_n_cuda = torch.matmul(torch.transpose(batch_pool_pred_lt_cuda, 1, 2), P_n_1_cuda)
            
            # M
            sum_P_n_1_cuda = torch.sum(P_n_1_cuda, dim=0)[None,None,:]
            sum_P_n_1_cuda[sum_P_n_1_cuda==0.0] = torch.finfo(torch.float).tiny

            # batch*class*M
            batch_P_n_1_P_n_div_P_n_1_sum_cuda = batch_P_n_1_P_n_cuda / sum_P_n_1_cuda
            
            # batch*class*M
            inside_log = (1.0/sample_num) * batch_P_n_1_P_n_cuda
            inside_log[inside_log==0.0] = torch.finfo(torch.float).tiny
            
            # batch*class*M
            batch_term1_cuda = batch_P_n_1_P_n_div_P_n_1_sum_cuda * torch.log(inside_log)

            # batch*(-1)
            batch_term1_cuda = batch_term1_cuda.view((batch_term1_cuda.size(0), -1))
            batch_term1_cuda = - torch.sum(batch_term1_cuda, dim=1) / (M)

            cate_prob = Categorical(batch_pool_pred_lt_cuda)
            batch_cur_entropy_cuda = cate_prob.entropy()
            batch_term2_cuda = torch.mean(batch_cur_entropy_cuda + sum_cond_entropy_cuda, dim=1) 
            batch_acq_sample_cuda = batch_term1_cuda - batch_term2_cuda
            acq_sample += list(batch_acq_sample_cuda.cpu().numpy())

            start += score_batch_size
 
        tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos = tmp_lt[i][0]
            if pos not in pos_lt:
                break
        pos_lt.append(pos)

        sample_prob = np.mean(pool_pred_lt[pos], axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        sample_prob_cuda = torch.from_numpy(sample_prob).to(device)
        yn_lt = torch.multinomial(sample_prob_cuda, M, replacement=True)
        
        pos_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[pos]).to(device)
        P_n_1_cuda = P_n_1_cuda*pos_pool_pred_lt_cuda[:,yn_lt]

        P_n_1_cuda[P_n_1_cuda==0.0] = torch.finfo(torch.float).tiny
       
        cate_prob = Categorical(pos_pool_pred_lt_cuda)
        cur_entropy_cuda = cate_prob.entropy()
        sum_cond_entropy_cuda += cur_entropy_cuda
    
    sel_pos_lt = [origin_pos_lt[pos] for pos in pos_lt]
    
    return sel_pos_lt



def acquire(model, pool, device, B=30, M=10000, sample_num=20, sampling_index=4, score_batch_size=40):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    origin_pos_lt = []
    pool_pred_lt = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            origin_pos_lt += list(pos.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)
    pool_size = pool.size()

    cluster2pos = pool.cluster2pos
    pos2cluster = pool.pos2cluster

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


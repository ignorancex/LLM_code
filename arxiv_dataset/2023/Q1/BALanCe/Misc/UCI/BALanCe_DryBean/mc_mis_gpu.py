import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical

import os

def delta_mis_batch_mc(model, pool, device, B=3, M=10000, sample_num=20, sampling_index=4, class_num=6, score_batch_size=40):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    pool_pred_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
        
    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)

    pool_size = pool.labels.shape[0]

    # K*1 at first
    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    pos_lt = []

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
   
    for b in range(sampling_index, B):
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

    return acq_sample, pos_lt





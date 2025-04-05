import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random

import os


def delta_mis_batch_mc(model, pool, device, B=3, M=100, sample_num=20, sampling_index=3, class_num=10):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False
            )
    
    pool_pred_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
        
    pool_pred_lt = np.array(pool_pred_lt)

    pool_size = pool.labels.shape[0]
    
    #pool_pred_lt_pre = pool_pred_lt
    pool_pred_lt_pre = np.ones((sample_num, 1))
    pos_lt = []

    score_batch_size = 500
    for b in range(sampling_index):
        acq_sample = []
        start = 0
        while start < pool_size:
            #print(f'small b: {b}, start: {start}')
            end = start + score_batch_size
            if end > pool_size:
                end = pool_size

            batch_pool_pred_lt = pool_pred_lt[start:end]
            batch_pool_pred_lt_cur = [[] for _ in range(end-start)]
            for i in range(start, end):
                for w in range(sample_num):
                    batch_pool_pred_lt_cur[i-start].append(np.outer(pool_pred_lt_pre[w], batch_pool_pred_lt[i-start][w]).reshape(-1))
            batch_pool_pred_lt_cur = np.array(batch_pool_pred_lt_cur)

            batch_p_y = np.mean(batch_pool_pred_lt_cur, axis=1)
            batch_entropy_p_y = entropy(batch_p_y, axis=1)

            batch_entropy_p_w_y = entropy(batch_pool_pred_lt_cur, axis=2)
            batch_mean_entropy_p_w_y = np.mean(batch_entropy_p_w_y, axis=1)

            batch_acq_sample = batch_entropy_p_y - batch_mean_entropy_p_w_y

            acq_sample += list(batch_acq_sample)
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
        
        pool_pred_lt_pre_tmp = []
        for w in range(sample_num):
            pool_pred_lt_pre_tmp.append(np.outer(pool_pred_lt_pre[w], pool_pred_lt[pos][w]).reshape(-1))
        pool_pred_lt_pre = np.array(pool_pred_lt_pre_tmp)

        start += score_batch_size

    #P_n_1 = pool_pred_lt_pre[pos]
    sum_cond_entropy = 0
    for pos in pos_lt:
        cur_entropy = entropy(pool_pred_lt[pos], axis=1)
        sum_cond_entropy += cur_entropy

    P_n_1 = np.ones((sample_num, M))
    for pos in pos_lt:
        sample_prob = np.mean(pool_pred_lt[pos], axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        yn_tmp = np.random.multinomial((M), sample_prob)
        yn_lt = []
        for yn in range(class_num):
            yn_lt += [yn for _ in range(yn_tmp[yn])]
        random.shuffle(yn_lt)
        yn_lt = np.array(yn_lt)
        P_n_1 = P_n_1*((pool_pred_lt[pos,:,yn_lt]).T)
        

    for b in range(sampling_index, B):
        start = 0
        acq_sample = []
        while start < pool_size:
            #print(f'large b: {b}, start: {start}')
            end = start + score_batch_size
            if end > pool_size:
                end = pool_size

            batch_pool_pred_lt = pool_pred_lt[start:end]
            batch_P_n_1_P_n = np.matmul(np.swapaxes(batch_pool_pred_lt, 1, 2), P_n_1)

            batch_P_n_1_P_n_div_P_n_1_sum = batch_P_n_1_P_n / np.sum(P_n_1, axis=0)
            batch_term1 = batch_P_n_1_P_n_div_P_n_1_sum * np.log((1.0/sample_num) * batch_P_n_1_P_n)
            batch_term1 = batch_term1.reshape((batch_term1.shape[0], -1))
            batch_term1 = - np.sum(batch_term1, axis=1) / (M)
                
            batch_cur_entropy = entropy(batch_pool_pred_lt, axis=2)
            batch_term2 = np.mean(batch_cur_entropy + sum_cond_entropy, axis=1)
        
            batch_acq_sample = batch_term1 - batch_term2
            acq_sample += list(batch_acq_sample)
            
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
        yn_tmp = np.random.multinomial((M), sample_prob) 
        yn_lt = []
        for yn in range(class_num):
            yn_lt += [yn for _ in range(yn_tmp[yn])]
        random.shuffle(yn_lt)
        yn_lt = np.array(yn_lt)
        P_n_1 = P_n_1*((pool_pred_lt[pos,:,yn_lt]).T)

        sum_cond_entropy += entropy(pool_pred_lt[pos], axis=1)

    return acq_sample, pos_lt



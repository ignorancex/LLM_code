import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def delta_batch_mc_eced(model, pool, hamming_pool, device, B=10, M=100, sample_num=20, sampling_index=3, hamming_dis_threshold=0.05, class_num=10):
    model.eval()
    dataset1 = MyDataset(pool)
    dataset2 = MyDataset(hamming_pool)
    dataloader1 = torch.utils.data.DataLoader(
            dataset1,
            batch_size=128,
            shuffle=False
            )
    dataloader2 = torch.utils.data.DataLoader(
            dataset2,
            batch_size=128,
            shuffle=False
            )
    
    pool_pred_lt1 = []
    pool_pred_lt2 = []

    hamming_pred_lt1 = []
    hamming_pred_lt2 = []

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #pred_lt = label_soft_pred.transpose(0,1)
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #label_soft_pred = label_soft_pred.transpose(0,1)
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt1 += list(pred_index)

    model.train(mode=False)
    model.eval()

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #pred_lt = label_soft_pred.transpose(0,1)
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #label_soft_pred = label_soft_pred.transpose(0,1)
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt2 += list(pred_index)

    pool_pred_lt1 = np.array(pool_pred_lt1, dtype=np.float32)
    pool_pred_lt2 = np.array(pool_pred_lt2, dtype=np.float32)

    hamming_pred_lt1 = np.array(hamming_pred_lt1, dtype=np.float32)
    hamming_pred_lt2 = np.array(hamming_pred_lt2, dtype=np.float32)

    hamming_dis_lt = [0.0 for i in range(sample_num)]
    for i in range(sample_num):
        dis = distance.hamming(hamming_pred_lt1[:,i], hamming_pred_lt2[:,i])
        if dis < hamming_dis_threshold:
            hamming_dis_lt[i] = 0
        else:
            hamming_dis_lt[i] = 1.0

    hamming_dis_lt = np.array(hamming_dis_lt, dtype=np.float32)
    
    pool_size = pool.labels.shape[0]
    
    max_lambda_g = [[] for i in range(pool_size)]
    max_lambda_g_prime = [[] for i in range(pool_size)]

    assert B > 1
    
    pool_pred_lt1_pre = pool_pred_lt1
    pool_pred_lt2_pre = pool_pred_lt2
    pos_lt = []

    for b in range(sampling_index):
        all_pred_lt = np.concatenate((pool_pred_lt1_pre, pool_pred_lt2_pre), axis=1)
        y_pred_lt = np.mean(all_pred_lt, axis=1)

        lambda_g_y = pool_pred_lt1_pre / np.expand_dims(np.amax(pool_pred_lt1_pre, axis=2), axis=2)
        lambda_g_prime_y = pool_pred_lt2_pre / np.expand_dims(np.amax(pool_pred_lt2_pre, axis=2), axis=2)

        total_lambda = (1.0 - lambda_g_y*lambda_g_prime_y)
        total_lambda = np.swapaxes(total_lambda, 1, 2)
        edge_cut = np.matmul(total_lambda, hamming_dis_lt)
        acq_sample = np.sum(y_pred_lt*edge_cut, axis=1)

        tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos = tmp_lt[i][0]
            if pos not in pos_lt:
                break
        pos_lt.append(pos)

        if b == sampling_index-1:
            break

        pool_pred_lt1_tmp = [[] for _ in range(pool_size)]
        for i in range(pool_size):
            for w in range(sample_num):
                pool_pred_lt1_tmp[i].append(np.outer(pool_pred_lt1_pre[pos][w], pool_pred_lt1[i][w]).reshape(-1))
        pool_pred_lt1_pre = np.array(pool_pred_lt1_tmp)

        pool_pred_lt2_tmp = [[] for _ in range(pool_size)]
        for i in range(pool_size):
            for w in range(sample_num):
                pool_pred_lt2_tmp[i].append(np.outer(pool_pred_lt2_pre[pos][w], pool_pred_lt2[i][w]).reshape(-1))
        pool_pred_lt2_pre = np.array(pool_pred_lt2_tmp)

    max_P_g = np.ones((sample_num,))
    max_P_g_prime = np.ones((sample_num,))   
    P_g_n_1 = np.ones((sample_num, 10**sampling_index))
    P_g_prime_n_1 = np.ones((sample_num, 10**sampling_index))
    for pos in pos_lt:
        tmp = np.concatenate((pool_pred_lt1[pos], pool_pred_lt2[pos]), axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        yn_tmp = np.random.multinomial((10**sampling_index), sample_prob)
        yn_lt = []
        for yn in range(class_num):
            yn_lt += [yn for _ in range(yn_tmp[yn])]
        random.shuffle(yn_lt)
        yn_lt = np.array(yn_lt)
        P_g_n_1 = P_g_n_1*((pool_pred_lt1[pos,:,yn_lt]).T)
        P_g_prime_n_1 = P_g_prime_n_1*((pool_pred_lt2[pos,:,yn_lt]).T)
        
        max_P_g = max_P_g * np.amax(pool_pred_lt1[pos], axis=1)
        max_P_g_prime = max_P_g_prime * np.amax(pool_pred_lt2[pos], axis=1)
    
    pool_pred_comb_lt = np.concatenate((pool_pred_lt1, pool_pred_lt2), axis=1)

    for b in range(sampling_index, B):
        # sample*k
        max_P_g = np.amax(pool_pred_lt1, axis=2) * max_P_g
        max_P_g_prime = np.amax(pool_pred_lt2, axis=2) * max_P_g_prime

        P_comb_n_1 = np.concatenate((P_g_n_1, P_g_prime_n_1), axis=0)
       
        P_comb_n_1_P_n = np.matmul(np.swapaxes(pool_pred_comb_lt, 1, 2), P_comb_n_1)/sample_num
        P_comb_n_1_P_n = np.swapaxes(P_comb_n_1_P_n, 1, 2)
        P_comb_n_1_div_P_n_1_sum = P_comb_n_1_P_n / (np.sum(P_comb_n_1, axis=0)[:, None])
        
        acq_sample = []
        for i in range(pool_size):
            # k*m*c
            P_g_n = np.matmul(P_g_n_1[:,:,np.newaxis], pool_pred_lt1[i][:,np.newaxis,:])
            P_g_prime_n = np.matmul(P_g_prime_n_1[:,:,np.newaxis], pool_pred_lt2[i][:,np.newaxis,:])
            discount = (1.0 - (P_g_n/(max_P_g[i][:,None,None]))*(P_g_prime_n/(max_P_g_prime[i][:,None,None])))
            #m*c 
            edge_cut = np.sum(discount*hamming_dis_lt[:,None, None], axis=0)
            acq_cur = sample_num*np.sum(P_comb_n_1_div_P_n_1_sum[i]*edge_cut)/(10**sampling_index)
            acq_sample.append(acq_cur)
        
        acq_sample = np.array(acq_sample)
        tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos_tmp = tmp_lt[i][0]
            if pos_tmp not in pos_lt:
                break
        pos = pos_tmp 
        pos_lt.append(pos)
    
        max_P_g = max_P_g[pos]
        max_P_g_prime = max_P_g_prime[pos]

        tmp = np.concatenate([pool_pred_lt1[pos], pool_pred_lt2[pos]], axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob) 
        yn_tmp = np.random.multinomial((10**sampling_index), sample_prob)
        yn_lt = []
        for yn in range(class_num):
            yn_lt += [yn for _ in range(yn_tmp[yn])]
        random.shuffle(yn_lt)
        P_g_n_1 = P_g_n_1*((pool_pred_lt1[pos,:, yn_lt]).T)
        P_g_prime_n_1 = P_g_prime_n_1 *((pool_pred_lt2[pos,:,yn_lt]).T)

    return acq_sample, pos_lt

import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os

def delta_mis(model, pool, device, sample_num=20):
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
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt)
    p_y = np.mean(pool_pred_lt, axis=1)
    entropy_p_y = entropy(p_y, axis=1)
   
    entropy_p_w_y = entropy(pool_pred_lt, axis=2)
    mean_entropy_p_w_y = np.mean(entropy_p_w_y, axis=1)

    delta_mis_sample = entropy_p_y - mean_entropy_p_w_y
    pos = np.argmax(delta_mis_sample)

    return delta_mis_sample, pos

def delta_mis_batch_enu(model, pool, device, B=4, sample_num=20, class_num=2):
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
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
        
    pool_pred_lt = np.array(pool_pred_lt)

    pool_size = pool.labels.shape[0]
    pool_pred_lt_pre = pool_pred_lt

    pos_lt = []

    for b in range(B):
        p_y = np.mean(pool_pred_lt_pre, axis=1)
        entropy_p_y = entropy(p_y, axis=1)
   
        entropy_p_w_y = entropy(pool_pred_lt_pre, axis=2)
        mean_entropy_p_w_y = np.mean(entropy_p_w_y, axis=1)

        delta_mis_sample = entropy_p_y - mean_entropy_p_w_y
        
        #print(delta_mis_sample)
        #pos = np.argmax(delta_mis_sample)
        tmp_lt = [(i, delta_mis_sample[i]) for i in range(len(delta_mis_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos = tmp_lt[i][0]
            if pos not in pos_lt:
                break
        pos_lt.append(pos)
        
        if b == B-1:
            break

        pool_pred_lt_tmp = [[] for _ in range(pool_size)]
        for i in range(pool_size):
            for w in range(sample_num):
                pool_pred_lt_tmp[i].append(np.outer(pool_pred_lt_pre[pos][w], pool_pred_lt[i][w]).reshape(-1))
        pool_pred_lt_pre = np.array(pool_pred_lt_tmp)
        
    return delta_mis_sample, pos_lt




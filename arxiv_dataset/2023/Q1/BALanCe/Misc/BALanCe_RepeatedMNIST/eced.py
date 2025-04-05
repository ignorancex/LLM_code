import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def delta_eced(model, pool, hamming_pool, device, sample_num=20, hamming_dis_threshold=0.05, class_num=10):
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

    hamming_dis_lt = [0.0 for i in range(sample_num)]
    pool_pred_lt1 = []
    pool_pred_lt2 = []

    hamming_pred_lt1 = []
    hamming_pred_lt2 = []

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            #pred_lt = label_soft_pred.transpose(0,1)
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
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
            label_soft_pred = pred.exp_()
            #pred_lt = label_soft_pred.transpose(0,1)
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            #label_soft_pred = label_soft_pred.transpose(0,1)
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt2 += list(pred_index)

    pool_pred_lt1 = np.array(pool_pred_lt1)
    pool_pred_lt2 = np.array(pool_pred_lt2)

    hamming_pred_lt1 = np.array(hamming_pred_lt1)
    hamming_pred_lt2 = np.array(hamming_pred_lt2)

    for i in range(sample_num):
        dis = distance.hamming(hamming_pred_lt1[:,i], hamming_pred_lt2[:,i])
        #hamming_dis_lt[i] = dis
        if dis < hamming_dis_threshold:
            hamming_dis_lt[i] = 0
        else:
            hamming_dis_lt[i] = 1.0

    all_pred_lt = np.concatenate((pool_pred_lt1, pool_pred_lt2), axis=1)
    y_pred_lt = np.mean(all_pred_lt, axis=1)

    lambda_g_y = pool_pred_lt1 / np.expand_dims(np.amax(pool_pred_lt1, axis=2), axis=2)
    lambda_g_prime_y = pool_pred_lt2 / np.expand_dims(np.amax(pool_pred_lt2, axis=2), axis=2)
  
    total_lambda = (1.0 - lambda_g_y*lambda_g_prime_y)
    total_lambda = np.swapaxes(total_lambda, 1, 2)
    edge_cut = np.matmul(total_lambda, hamming_dis_lt)
    acq_sample = np.sum(y_pred_lt*edge_cut, axis=1)

    pos = np.argmax(acq_sample)
    return acq_sample, pos


def delta_batch_eced(model, pool, hamming_pool, device, B=2, sample_num=20, hamming_dis_threshold=0.05, class_num=10):
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
    
    hamming_dis_lt = [0.0 for i in range(sample_num)]
    pool_pred_lt1 = []
    pool_pred_lt2 = []

    hamming_pred_lt1 = []
    hamming_pred_lt2 = []
   
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt1 += list(pred_index)

    model.train(mode=False)
    model.eval()

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt2 += list(pred_index)

    pool_pred_lt1 = np.array(pool_pred_lt1)
    pool_pred_lt2 = np.array(pool_pred_lt2)

    hamming_pred_lt1 = np.array(hamming_pred_lt1)
    hamming_pred_lt2 = np.array(hamming_pred_lt2)

    for i in range(sample_num):
        dis = distance.hamming(hamming_pred_lt1[i], hamming_pred_lt2[i])
        if dis < hamming_dis_threshold:
            hamming_dis_lt[i] = 0
        else:
            hamming_dis_lt[i] = 1.0

    assert B > 1
    pool_size = pool.labels.shape[0]

    pos_lt = []
    
    pool_pred_lt1_pre = pool_pred_lt1
    pool_pred_lt2_pre = pool_pred_lt2

    for b in range(B):
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
        
        if b == B-1:
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
        
    return acq_sample, pos_lt


def delta_pseudo_eced(model, pool, hamming_pool, device, B=10, sample_num=20, hamming_dis_threshold=0.05, class_num=10):
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

    hamming_dis_lt = [0.0 for i in range(sample_num)]
    pool_pred_lt1 = []
    pool_pred_lt2 = []

    hamming_pred_lt1 = []
    hamming_pred_lt2 = []
   
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt1 += list(pred_index)

    model.train(mode=False)
    model.eval()

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt2 += list(pred_index)

    pool_pred_lt1 = np.array(pool_pred_lt1)
    pool_pred_lt2 = np.array(pool_pred_lt2)

    hamming_pred_lt1 = np.array(hamming_pred_lt1)
    hamming_pred_lt2 = np.array(hamming_pred_lt2)

    for i in range(sample_num):
        dis = distance.hamming(hamming_pred_lt1[:,i], hamming_pred_lt2[:,i])
        #hamming_dis_lt[i] = dis
        if dis < hamming_dis_threshold:
            hamming_dis_lt[i] = 0
        else:
            hamming_dis_lt[i] = 1.0

    all_pred_lt = np.concatenate((pool_pred_lt1, pool_pred_lt2), axis=1)
    y_pred_lt = np.mean(all_pred_lt, axis=1)

    lambda_g_y = pool_pred_lt1 / np.expand_dims(np.amax(pool_pred_lt1, axis=2), axis=2)
    lambda_g_prime_y = pool_pred_lt2 / np.expand_dims(np.amax(pool_pred_lt2, axis=2), axis=2)
  
    total_lambda = (1.0 - lambda_g_y*lambda_g_prime_y)
    total_lambda = np.swapaxes(total_lambda, 1, 2)
    edge_cut = np.matmul(total_lambda, hamming_dis_lt)
    acq_sample = np.sum(y_pred_lt*edge_cut, axis=1)

    pos_lt = []
    tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
    tmp_lt.sort(key=lambda x:x[1], reverse=True)
    for i in range(B):
        pos = tmp_lt[i][0]
        pos_lt.append(pos)

    return acq_sample, pos_lt
 


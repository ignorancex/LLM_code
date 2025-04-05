import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def power_balance(model, pool, hamming_pool, device, B=10, sample_num=20, hamming_dis_threshold=0.05, class_num=10, coldness=1.0):
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
    acq_sample = np.sum(y_pred_lt*edge_cut, axis=1)/sample_num
    
    print('max acq sample value')
    print(np.max(acq_sample))
    #pos = np.argmax(acq_sample)
    #return acq_sample, pos
    
    acq_sample[acq_sample <= 0] = 1e-6
    log_sample_acq = np.log(acq_sample)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=log_sample_acq.shape)
    index_lt = np.argpartition(gumbel, -B)[-B:]
    
    return list(index_lt)






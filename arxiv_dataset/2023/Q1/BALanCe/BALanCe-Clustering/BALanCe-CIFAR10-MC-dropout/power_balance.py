import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def acquire(model, pool, hamming_pool, device, coldness=1.0, B=10, sample_num=20, hamming_dis_threshold=0.05, class_num=10, score_batch_size=20):
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
    origin_pos_lt = []

    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())
            origin_pos_lt += list(pos.data.cpu().numpy())

        for b_id, (pos, x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt1 += list(pred_index)

    model.train(mode=False)
    model.eval()

    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader1):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (pos, x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            label_soft_pred = label_soft_pred.data.cpu().numpy()
            pred_index = np.argmax(label_soft_pred, axis=2)
            hamming_pred_lt2 += list(pred_index)

    del model

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
    hamming_dis_lt_cuda = torch.from_numpy(hamming_dis_lt).to(device)

    pool_size = len(origin_pos_lt)
    
    pool_pred_lt1_pre = np.ones((pool_pred_lt1.shape[1], 1), dtype=np.float32)
    pool_pred_lt2_pre = np.ones((pool_pred_lt2.shape[1], 1), dtype=np.float32)

    pool_pred_lt1_pre_cuda = torch.from_numpy(pool_pred_lt1_pre).to(device)
    pool_pred_lt2_pre_cuda = torch.from_numpy(pool_pred_lt2_pre).to(device)

    start = 0
    acq_sample = []
    while start < pool_size:
        end = start + score_batch_size
        if end > pool_size:
            end = pool_size

        # batch*K*C
        batch_pool_pred_lt1 = pool_pred_lt1[start:end]
        batch_pool_pred_lt2 = pool_pred_lt2[start:end]
        
        # batch*K*C
        batch_pool_pred_lt1_cuda = torch.from_numpy(batch_pool_pred_lt1).to(device)
        batch_pool_pred_lt2_cuda = torch.from_numpy(batch_pool_pred_lt2).to(device)
        
        # batch*K*C^b
        # batch_pool_pred_lt1_cur_cuda = torch.matmul(batch_pool_pred_lt1_cuda[:,:,:,None], pool_pred_lt1_pre_cuda[None,:,None,:])
        # batch_pool_pred_lt2_cur_cuda = torch.matmul(batch_pool_pred_lt2_cuda[:,:,:,None], pool_pred_lt2_pre_cuda[None,:,None,:])
        batch_pool_pred_lt1_cur_cuda = torch.matmul( pool_pred_lt1_pre_cuda[None,:,:,None], batch_pool_pred_lt1_cuda[:,:,None,:])
        batch_pool_pred_lt2_cur_cuda = torch.matmul( pool_pred_lt2_pre_cuda[None,:,:,None], batch_pool_pred_lt2_cuda[:,:,None,:])
        
        tmp_size = batch_pool_pred_lt1_cur_cuda.size()
        batch_pool_pred_lt1_cur_cuda = batch_pool_pred_lt1_cur_cuda.view((tmp_size[0], tmp_size[1], -1))
        batch_pool_pred_lt2_cur_cuda = batch_pool_pred_lt2_cur_cuda.view((tmp_size[0], tmp_size[1], -1))
        
        # batch*2K*C^b
        batch_all_pred_lt_cuda = torch.cat((batch_pool_pred_lt1_cur_cuda, batch_pool_pred_lt2_cur_cuda), dim=1)
        # batch*C^b
        batch_y_pred_lt_cuda = torch.mean(batch_all_pred_lt_cuda, dim=1)

        # batch*K
        max_batch_pool_pred_lt1_cur_cuda = torch.amax(batch_pool_pred_lt1_cur_cuda, dim=2)
        max_batch_pool_pred_lt2_cur_cuda = torch.amax(batch_pool_pred_lt2_cur_cuda, dim=2)
        
        # batch*K*C^b
        batch_lambda_g_y_cuda = batch_pool_pred_lt1_cur_cuda / max_batch_pool_pred_lt1_cur_cuda[:,:,None]
        batch_lambda_g_prime_y_cuda = batch_pool_pred_lt2_cur_cuda / max_batch_pool_pred_lt2_cur_cuda[:,:,None]

        # batch*K*C^b
        batch_total_lambda_cuda = (1.0 - batch_lambda_g_y_cuda*batch_lambda_g_prime_y_cuda)
        
        # batch*C^b
        batch_edge_cut = torch.matmul(hamming_dis_lt_cuda[None,None,:], batch_total_lambda_cuda ) 
        batch_edge_cut = torch.squeeze(batch_edge_cut, dim=1)

        # batch
        batch_acq_sample_cuda = torch.sum(batch_y_pred_lt_cuda*batch_edge_cut, dim=1)/sample_num

        acq_sample += list(batch_acq_sample_cuda.cpu().numpy())

        start += score_batch_size
   
    log_sample_acq = np.log(acq_sample)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=log_sample_acq.shape)
    index_lt = np.argpartition(gumbel, -B)[-B:]
    pos_lt = []
    for index in index_lt:
        pos_lt.append(origin_pos_lt[index])

    return pos_lt

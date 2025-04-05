import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def delta_batch_mc_eced(model, pool, hamming_pool, device, B=10, M=10000, sample_num=20, sampling_index=4, hamming_dis_threshold=0.05, class_num=10, score_batch_size=20):
    model.eval()
    dataset1 = MyDataset(pool)
    dataset2 = MyDataset(hamming_pool)
    dataloader1 = torch.utils.data.DataLoader(
            dataset1,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    dataloader2 = torch.utils.data.DataLoader(
            dataset2,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
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

    pool_size = pool.labels.shape[0]
    
    pool_pred_lt1_pre = np.ones((pool_pred_lt1.shape[1], 1), dtype=np.float32)
    pool_pred_lt2_pre = np.ones((pool_pred_lt2.shape[1], 1), dtype=np.float32)

    pool_pred_lt1_pre_cuda = torch.from_numpy(pool_pred_lt1_pre).to(device)
    pool_pred_lt2_pre_cuda = torch.from_numpy(pool_pred_lt2_pre).to(device)

    pos_lt = []
    
    for b in range(sampling_index):
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
        
        assert len(acq_sample) == pool_size
        
        tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos = tmp_lt[i][0]
            if pos not in pos_lt:
                break
        pos_lt.append(pos)

        if b == sampling_index-1:
            break
        
        pos_pool_pred_lt1_cuda = torch.from_numpy(pool_pred_lt1[pos]).to(device)
        pool_pred_lt1_pre_cuda = torch.matmul(pool_pred_lt1_pre_cuda[:,:,None], pos_pool_pred_lt1_cuda[:,None,:])
        tmp_size = pool_pred_lt1_pre_cuda.size()
        pool_pred_lt1_pre_cuda = pool_pred_lt1_pre_cuda.view((tmp_size[0],-1))
        
        pos_pool_pred_lt2_cuda = torch.from_numpy(pool_pred_lt2[pos]).to(device)
        pool_pred_lt2_pre_cuda = torch.matmul(pool_pred_lt2_pre_cuda[:,:,None], pos_pool_pred_lt2_cuda[:,None,:])
        tmp_size = pool_pred_lt2_pre_cuda.size()
        pool_pred_lt2_pre_cuda = pool_pred_lt2_pre_cuda.view((tmp_size[0],-1))
        
    max_P_g_cuda = torch.ones((sample_num,), device=device)
    max_P_g_prime_cuda = torch.ones((sample_num,), device=device)
     
    P_g_n_1_cuda = torch.ones((sample_num, M), device=device)
    P_g_prime_n_1_cuda = torch.ones((sample_num, M), device=device)

    for pos in pos_lt:
        tmp = np.concatenate((pool_pred_lt1[pos], pool_pred_lt2[pos]), axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        sample_prob_cuda = torch.from_numpy(sample_prob)
        
        yn_lt = torch.multinomial(sample_prob_cuda, M, replacement=True)
        pos_pool_pred_lt1 = pool_pred_lt1[pos]
        pos_pool_pred_lt2 = pool_pred_lt2[pos]
        pos_pool_pred_lt1_cuda = torch.from_numpy(pos_pool_pred_lt1).to(device)
        pos_pool_pred_lt2_cuda = torch.from_numpy(pos_pool_pred_lt2).to(device)
        
        P_g_n_1_cuda = P_g_n_1_cuda*pos_pool_pred_lt1_cuda[:,yn_lt]
        P_g_prime_n_1_cuda = P_g_prime_n_1_cuda*pos_pool_pred_lt2_cuda[:,yn_lt]
        
        max_P_g_cuda = max_P_g_cuda*torch.amax(pos_pool_pred_lt1_cuda, dim=1)
        max_P_g_prime_cuda = max_P_g_prime_cuda*torch.amax(pos_pool_pred_lt2_cuda, dim=1)
    
    pool_pred_comb_lt = np.concatenate((pool_pred_lt1, pool_pred_lt2), axis=1)
    
    for b in range(sampling_index, B):
        # sample*k
        # print(f'larger b: {b}')
        acq_sample = []
        start = 0
        while start < pool_size:
            #print(f'\tlarger b: {b}, start: {start}')
            if start + score_batch_size < pool_size:
                end = start + score_batch_size
            else:
                end = pool_size
            
            pool_pred_lt1_mini_cuda = torch.from_numpy(pool_pred_lt1[start:end]).to(device)
            # batch*K
            max_P_g_mini_cuda = torch.amax(pool_pred_lt1_mini_cuda, dim=2) * max_P_g_cuda[None,:]
            
            pool_pred_lt2_mini_cuda = torch.from_numpy(pool_pred_lt2[start:end]).to(device)
            # batch*K
            max_P_g_prime_mini_cuda = torch.amax(pool_pred_lt2_mini_cuda, dim=2) * max_P_g_prime_cuda[None,:]

            # 2K*M
            P_comb_n_1_cuda = torch.cat((P_g_n_1_cuda, P_g_prime_n_1_cuda), dim=0)
            # 2K*C
            pool_pred_comb_lt_mini_cuda = torch.from_numpy(pool_pred_comb_lt[start:end]).to(device)
            # batch*C*M
            P_comb_n_1_P_n_mini_cuda = torch.matmul(torch.transpose(pool_pred_comb_lt_mini_cuda, 1, 2), P_comb_n_1_cuda)/sample_num
            # batch*M*C
            P_comb_n_1_P_n_mini_cuda = torch.transpose(P_comb_n_1_P_n_mini_cuda, 1, 2)
            
            # M
            fac_2 = torch.sum(P_comb_n_1_cuda, dim=0)[:, None]
            fac_2[fac_2==0.0] = torch.finfo(torch.float).tiny 

            # batch*M*C
            #P_comb_n_1_div_P_n_1_sum_mini_cuda = P_comb_n_1_P_n_mini_cuda / (torch.sum(P_comb_n_1_cuda, dim=0)[:, None])
            P_comb_n_1_div_P_n_1_sum_mini_cuda = P_comb_n_1_P_n_mini_cuda / fac_2
            
            # batch*K*M*C
            P_g_n_mini_cuda = torch.matmul(P_g_n_1_cuda[None,:,:,None], pool_pred_lt1_mini_cuda[:,:,None,:])
            P_g_prime_n_mini_cuda = torch.matmul(P_g_prime_n_1_cuda[None,:,:,None], pool_pred_lt2_mini_cuda[:,:,None,:])
            
            # batch*K*M*C
            discount_mini_cuda = 1.0 - (P_g_n_mini_cuda/(max_P_g_mini_cuda[:,:,None,None]))*(P_g_prime_n_mini_cuda/(max_P_g_prime_mini_cuda[:,:,None,None]))
            # batch*M*C
            edge_cut_mini_cuda = torch.sum(discount_mini_cuda * hamming_dis_lt_cuda[None,:,None,None], dim=1)
            # batch
            acq_sample_mini_cuda = torch.sum(P_comb_n_1_div_P_n_1_sum_mini_cuda * edge_cut_mini_cuda, dim=(1,2))/(M)

            acq_sample += list(acq_sample_mini_cuda.cpu().numpy())
            #print(f'mini_start:{mini_start}, acq_sample: {acq_sample}') 
            start += score_batch_size
        
        tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
        tmp_lt.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(tmp_lt)):
            pos_tmp = tmp_lt[i][0]
            if pos_tmp not in pos_lt:
                break
        pos = pos_tmp 
        pos_lt.append(pos)

        pos_pool_pred_lt1_cuda = torch.from_numpy(pool_pred_lt1[pos]).to(device)
        max_P_g_cuda = torch.amax(pos_pool_pred_lt1_cuda, dim=1)*max_P_g_cuda
        
        pos_pool_pred_lt2_cuda = torch.from_numpy(pool_pred_lt2[pos]).to(device)
        max_P_g_prime_cuda = torch.amax(pos_pool_pred_lt2_cuda, dim=1)*max_P_g_prime_cuda
        
        tmp = np.concatenate((pool_pred_lt1[pos], pool_pred_lt2[pos]), axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        sample_prob_cuda = torch.from_numpy(sample_prob)
        
        yn_lt = torch.multinomial(sample_prob_cuda, M, replacement=True)
        pos_pool_pred_lt1 = pool_pred_lt1[pos]
        pos_pool_pred_lt2 = pool_pred_lt2[pos]
        pos_pool_pred_lt1_cuda = torch.from_numpy(pos_pool_pred_lt1).to(device)
        pos_pool_pred_lt2_cuda = torch.from_numpy(pos_pool_pred_lt2).to(device)
        
        P_g_n_1_cuda = P_g_n_1_cuda*pos_pool_pred_lt1_cuda[:,yn_lt]
        P_g_prime_n_1_cuda = P_g_prime_n_1_cuda*pos_pool_pred_lt2_cuda[:,yn_lt]
        
    return acq_sample, pos_lt

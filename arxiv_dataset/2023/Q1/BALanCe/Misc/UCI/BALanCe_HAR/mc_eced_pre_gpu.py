import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random

def delta_batch_mc_eced(model, pool, hamming_pool, device, B=10, M=10000, sample_num=20, sampling_index=4, hamming_dis_threshold=0.05, class_num=10):
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
            pool_pred_lt1 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
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
            pool_pred_lt2 += list(label_soft_pred.data.cpu().numpy())

        for b_id, (x, label) in enumerate(dataloader2):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
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
    
    #pool_pred_lt1_pre = pool_pred_lt1
    #pool_pred_lt2_pre = pool_pred_lt2
    pool_pred_lt1_pre = np.ones((pool_pred_lt1.shape[1], 1))
    pool_pred_lt2_pre = np.ones((pool_pred_lt2.shape[1], 1))

    pos_lt = []
    
    score_batch_size = 500
    for b in range(sampling_index):
        print(f'smaller b: {b}')
        start = 0
        acq_sample = []
        while start < pool_size:
            print(f'\tsmaller b: {b}, start: {start}')
            end = start + score_batch_size
            if end > pool_size:
                end = pool_size

            batch_pool_pred_lt1 = pool_pred_lt1[start:end]
            batch_pool_pred_lt2 = pool_pred_lt2[start:end]

            # here
            batch_pool_pred_lt1_cur = [[] for _ in range(end-start)]
            batch_pool_pred_lt2_cur = [[] for _ in range(end-start)]
            
            for i in range(start, end):
                for w in range(sample_num):
                    batch_pool_pred_lt1_cur[i-start].append(np.outer(pool_pred_lt1_pre[w], batch_pool_pred_lt1[i-start][w]).reshape(-1))
            batch_pool_pred_lt1_cur = np.array(batch_pool_pred_lt1_cur)

            for i in range(start, end):
                for w in range(sample_num):
                    batch_pool_pred_lt2_cur[i-start].append(np.outer(pool_pred_lt2_pre[w], batch_pool_pred_lt2[i-start][w]).reshape(-1))
            batch_pool_pred_lt2_cur = np.array(batch_pool_pred_lt2_cur)

            batch_all_pred_lt = np.concatenate((batch_pool_pred_lt1_cur, batch_pool_pred_lt2_cur), axis=1) 
            batch_y_pred_lt = np.mean(batch_all_pred_lt, axis=1)

            batch_lambda_g_y = batch_pool_pred_lt1_cur / np.expand_dims(np.amax(batch_pool_pred_lt1_cur, axis=2), axis=2)
            batch_lambda_g_prime_y = batch_pool_pred_lt2_cur / np.expand_dims(np.amax(batch_pool_pred_lt2_cur, axis=2), axis=2)

            batch_total_lambda = (1.0 - batch_lambda_g_y*batch_lambda_g_prime_y)
            batch_total_lambda = np.swapaxes(batch_total_lambda, 1, 2)
            batch_edge_cut = np.matmul(batch_total_lambda, hamming_dis_lt)
            batch_acq_sample = np.sum(batch_y_pred_lt*batch_edge_cut, axis=1)

            acq_sample += list(batch_acq_sample)
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

        pool_pred_lt1_tmp = []
        for w in range(sample_num):
            pool_pred_lt1_tmp.append(np.outer(pool_pred_lt1_pre[w], pool_pred_lt1[pos][w]).reshape(-1))
        pool_pred_lt1_pre = np.array(pool_pred_lt1_tmp)

        pool_pred_lt2_tmp = []
        for w in range(sample_num):
            pool_pred_lt2_tmp.append(np.outer(pool_pred_lt2_pre[w], pool_pred_lt2[pos][w]).reshape(-1))
        pool_pred_lt2_pre = np.array(pool_pred_lt2_tmp)


    max_P_g = np.ones((sample_num,))
    max_P_g_prime = np.ones((sample_num,))

    P_g_n_1 = np.ones((sample_num, M))
    P_g_prime_n_1 = np.ones((sample_num, M))
    for pos in pos_lt:
        tmp = np.concatenate((pool_pred_lt1[pos], pool_pred_lt2[pos]), axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob)
        yn_tmp = np.random.multinomial(M, sample_prob)
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
        print(f'larger b: {b}')
        acq_sample = []
        start = 0
        while start < pool_size:
            print(f'\tlarger b: {b}, start: {start}')
            if start + score_batch_size < pool_size:
                end = start + score_batch_size
            else:
                end = pool_size

            max_P_g_mini = np.amax(pool_pred_lt1[start:end], axis=2) * max_P_g
            max_P_g_prime_mini = np.amax(pool_pred_lt2[start:end], axis=2) * max_P_g_prime 
            P_comb_n_1 = np.concatenate((P_g_n_1, P_g_prime_n_1), axis=0)
            
            print(f'start P_comb_n_1_div_P_n_1_sum_mini')
            # sample*m*c
            P_comb_n_1_P_n_mini = np.matmul(np.swapaxes(pool_pred_comb_lt[start:end], 1, 2), P_comb_n_1)/sample_num
            P_comb_n_1_P_n_mini = np.swapaxes(P_comb_n_1_P_n_mini, 1, 2)
            P_comb_n_1_div_P_n_1_sum_mini = P_comb_n_1_P_n_mini / (np.sum(P_comb_n_1, axis=0)[:, None])
            print(f'end P_comb_n_1_div_P_n_1_sum_mini')

            print(f'start edge cut')
            # sample*k*m*c
            print(f'\t1')
            P_g_n_mini = np.matmul(P_g_n_1[np.newaxis,:,:,np.newaxis], pool_pred_lt1[start:end,:,np.newaxis,:])
            print(f'\t2')
            P_g_prime_n_mini = np.matmul(P_g_prime_n_1[np.newaxis,:,:,np.newaxis], pool_pred_lt2[start:end,:,np.newaxis,:])
            print(f'\t3')
            discount_mini = 1.0 - (P_g_n_mini/(max_P_g_mini[:,:,None,None]))*(P_g_prime_n_mini/(max_P_g_prime_mini[:,:,None,None]))
            # sample*m*c
            print(f'\t4')
            edge_cut_mini = np.sum(discount_mini * hamming_dis_lt[np.newaxis,:,None,None], axis=1)
            print(f'\t5')
            acq_sample_mini = np.sum(P_comb_n_1_div_P_n_1_sum_mini*edge_cut_mini, axis=(1,2))/(M)
            print(f'end edge cut')

            acq_sample += list(acq_sample_mini)
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

        #max_P_g = max_P_g[pos]
        #max_P_g_prime = max_P_g_prime[pos]
        max_P_g = np.amax(pool_pred_lt1[pos], axis=1)*max_P_g
        max_P_g_prime = np.amax(pool_pred_lt2[pos], axis=1)*max_P_g_prime
        
        print(f'start sampling')
        tmp = np.concatenate([pool_pred_lt1[pos], pool_pred_lt2[pos]], axis=0)
        sample_prob = np.mean(tmp, axis=0).astype(np.float64)
        sample_prob = sample_prob / np.sum(sample_prob) 
        yn_tmp = np.random.multinomial((M), sample_prob)
        yn_lt = []
        for yn in range(class_num):
            yn_lt += [yn for _ in range(yn_tmp[yn])]
        random.shuffle(yn_lt)
        P_g_n_1 = P_g_n_1*((pool_pred_lt1[pos,:, yn_lt]).T)
        P_g_prime_n_1 = P_g_prime_n_1*((pool_pred_lt2[pos,:,yn_lt]).T)
        print(f'end sampling')
    
    return acq_sample, pos_lt

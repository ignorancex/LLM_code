import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical
from multiprocessing import Process
import os
import multiprocessing

def get_If(sample_acq, pair_acq):
    sample_num = len(sample_acq)
    If = [[0.0]*sample_num for _ in range(sample_num)]
    for i in range(sample_num):
        for j in range(sample_num):
            If[i][j] = sample_acq[i] + sample_acq[j] - pair_acq[i][j]
    return If

def downsample(pool_pred_lt, sample_acq, downsample_num, coldness=1.0):
    p_sample_acq = np.array(sample_acq)
    p_sample_acq = coldness*p_sample_acq
    p_sample_acq = np.exp(p_sample_acq)
    p_sample_acq = p_sample_acq/(np.sum(p_sample_acq))
    index_lt = np.random.choice(len(pool_pred_lt), downsample_num, replace=False, p=p_sample_acq)
    #index_lt = [i for i in range(downsample_num)]
    downsample_pool_pred_lt = pool_pred_lt[index_lt]
    return downsample_pool_pred_lt, index_lt

def get_single_acq(pool_pred_lt, sample_num, device, score_batch_size):
    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    print(f'cal singleton value')
    acq_sample = []
    start = 0
    downsample_num = len(pool_pred_lt)
    while start < downsample_num:
        end = start + score_batch_size
        if end > downsample_num:
            end = downsample_num
        
        # batch*K*C
        batch_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[start:end]).to(device)
        batch_pool_pred_lt_cur_cuda = torch.matmul(pool_pred_lt_pre_cuda[None,:,:,None], batch_pool_pred_lt_cuda[:,:,None,:])
        tmp_size = batch_pool_pred_lt_cur_cuda.size()
        # batch*K*C^b
        batch_pool_pred_lt_cur_cuda = batch_pool_pred_lt_cur_cuda.view((tmp_size[0], tmp_size[1], -1))
        # batch*C^b
        batch_p_y_cuda = torch.mean(batch_pool_pred_lt_cur_cuda, dim=1)
        cate_prob = Categorical(batch_p_y_cuda)
        batch_entropy_p_y_cuda = cate_prob.entropy()
        
        # batch*K
        cate_prob = Categorical(batch_pool_pred_lt_cur_cuda)
        batch_entropy_p_w_y_cuda = cate_prob.entropy()
        batch_mean_entropy_p_w_y_cuda = torch.mean(batch_entropy_p_w_y_cuda, dim=1)

        batch_acq_sample_cuda = batch_entropy_p_y_cuda - batch_mean_entropy_p_w_y_cuda

        acq_sample += list(batch_acq_sample_cuda.cpu().numpy())
        start += score_batch_size
    
    return acq_sample

def get_acq_pos_other(pos, pool_pred_lt, device, sample_num=20, score_batch_size=40):
    pos_pred_lt_cuda = torch.from_numpy(pool_pred_lt[pos]).to(device)
    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    pool_pred_lt_pre_cuda = torch.matmul(pool_pred_lt_pre_cuda[:,:,None], pos_pred_lt_cuda[:,None,:])
    pool_pred_lt_pre_cuda = pool_pred_lt_pre_cuda.view((sample_num, -1))

    acq_sample_other_lt = []
    start = 0
    downsample_num = len(pool_pred_lt)
    while start < downsample_num:
        #print(f'small b: {b}, start: {start}')
        end = start + score_batch_size
        if end > downsample_num:
            end = downsample_num

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

        acq_sample_other_lt += list(batch_acq_sample_cuda.cpu().numpy())

        start += score_batch_size

    return acq_sample_other_lt

def get_single_pair_f(pool_pred_lt, sample_num, device, score_batch_size):
    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    print(f'cal singleton value')
    acq_sample = []
    start = 0
    downsample_num = len(pool_pred_lt)
    while start < downsample_num:
        end = start + score_batch_size
        if end > downsample_num:
            end = downsample_num
        
        # batch*K*C
        batch_pool_pred_lt_cuda = torch.from_numpy(pool_pred_lt[start:end]).to(device)
        batch_pool_pred_lt_cur_cuda = torch.matmul(pool_pred_lt_pre_cuda[None,:,:,None], batch_pool_pred_lt_cuda[:,:,None,:])
        tmp_size = batch_pool_pred_lt_cur_cuda.size()
        # batch*K*C^b
        batch_pool_pred_lt_cur_cuda = batch_pool_pred_lt_cur_cuda.view((tmp_size[0], tmp_size[1], -1))
        # batch*C^b
        batch_p_y_cuda = torch.mean(batch_pool_pred_lt_cur_cuda, dim=1)
        cate_prob = Categorical(batch_p_y_cuda)
        batch_entropy_p_y_cuda = cate_prob.entropy()
        
        # batch*K
        cate_prob = Categorical(batch_pool_pred_lt_cur_cuda)
        batch_entropy_p_w_y_cuda = cate_prob.entropy()
        batch_mean_entropy_p_w_y_cuda = torch.mean(batch_entropy_p_w_y_cuda, dim=1)

        batch_acq_sample_cuda = batch_entropy_p_y_cuda - batch_mean_entropy_p_w_y_cuda

        acq_sample += list(batch_acq_sample_cuda.cpu().numpy())
        start += score_batch_size

   
    '''
    fout = open('single_info.txt', 'w')
    fout.write('pos\tinfo\n')
    for i in range(downsample_num):
        fout.write(f'{i}\t{acq_sample[i]}\n')
    fout.close()

    print(f'cal pair values')
    fout = open('pair_mut_info.txt', 'w')
    fout.write('pos\t')
    for i in range(downsample_num):
        fout.write(f'{i}\t')
    fout.write('\n')
    fout.write(f'{pos}\t')
        for j in range(downsample_num):
            fout.write(f'{pos_result[j]}\t')
        fout.write('\n')

    fout.close()
    '''
    pair_result = []
    for pos in range(downsample_num):
        pos_result = get_acq_pos_other(pos, pool_pred_lt, device, sample_num=sample_num, score_batch_size=score_batch_size)
        pair_result.append(pos_result)


    for i in range(len(acq_sample)):
        pair_result[i][i] = acq_sample[i]

    return acq_sample, pair_result

def assign_cluster(pool_pred_lt, mu_lt, sample_acq, sample_num, device, score_batch_size):
    centroid2other_lt = []
    for pos in mu_lt:
        centroid2other_lt.append(get_acq_pos_other(pos, pool_pred_lt, device, sample_num=sample_num, score_batch_size=score_batch_size))
        for j in range(len(pool_pred_lt)):
            if j != pos:
                centroid2other_lt[-1][j] = sample_acq[pos] + sample_acq[j] - centroid2other_lt[-1][j]
            else:
                centroid2other_lt[-1][j] = sample_acq[pos] 

    centroid2other_lt = np.array(centroid2other_lt)

    argmax_centroid = np.argmax(centroid2other_lt.T, axis=1)
    cluster2pos = {}
    for i in range(len(mu_lt)):
        pos = mu_lt[i]
        cluster2pos[i] = [pos]
    
    print('initial cluster2pos')
    print(cluster2pos)
    for i in range(len(pool_pred_lt)):
        if i in mu_lt:
            print(f'\t {i} in mu_lt')
        else:
            cluster = argmax_centroid[i]
            cluster2pos[cluster].append(i)
    
    print('check pos')
    check_pos2cluster = {}
    for cluster_label in cluster2pos:
        for pos in cluster2pos[cluster_label]:
            if pos in check_pos2cluster:
                print(f'cluster: {cluster_label}, pos: {pos} already exists')
                sys.exit()
            else:
                check_pos2cluster[pos] = cluster_label

    return cluster2pos, argmax_centroid

def acq_driven_clustering(sample_acq, If, cluster_num, coldness=1.0):
    '''
    index_lt = [i for i in range(len(sample_acq))]
    random.shuffle(index_lt)
    assert len(sample_acq) > cluster_num
    seed_lt = index_lt[:cluster_num]
    '''
    sample_acq = np.array(sample_acq)
    If = np.array(If)
    p_sample_acq = sample_acq
    p_sample_acq[p_sample_acq<0.0] = 0.0
    p_sample_acq *= coldness
    p_sample_acq = np.exp(p_sample_acq)
    p_sample_acq = p_sample_acq/np.sum(p_sample_acq)
    seed_lt = np.random.choice(len(sample_acq), cluster_num, replace=False, p=p_sample_acq)
   
    iteration = 0
    while True:
        iteration += 1
        print(f'Iteration: {iteration}')
        sub_If = If[:, seed_lt]
        print(f'seed_lt: {seed_lt}')
        argmax_index = np.argmax(sub_If, axis=1)
        #print(argmax_index.shape)
        cluster_lt = [[seed_lt[i]] for i in range(len(seed_lt))]

        for i in range(len(sample_acq)):
            if i not in seed_lt:
                cluster_lt[argmax_index[i]].append(i)

        #print(cluster_lt)
        
        cluster_element_val = [[] for i in range(len(cluster_lt))]
        for i in range(len(cluster_lt)):
            if len(cluster_lt[i]) == 1:
                cluster_element_val[i] = [(cluster_lt[i][0], sample_acq[seed_lt[i]])]

            cluster_element_val[i] = []
            for j in cluster_lt[i]:
                sum4j = 0.0
                for k in cluster_lt[i]:
                    if k != j:
                        sum4j += If[j][k]
                cluster_element_val[i].append((j, sum4j))

        mu_lt = []
        for i in range(len(cluster_lt)):
            if len(cluster_element_val[i]) == 1:
                mu_lt.append(cluster_element_val[i][0][0])
            else:
                max_tuple = max(cluster_element_val[i], key=lambda x:x[1])
                max_j = max_tuple[0]
                mu_lt.append(max_j)

        print(f'mu_lt:{mu_lt}') 
        mu_lt = np.array(mu_lt)
    
        if np.all(mu_lt == seed_lt):
            break
        seed_lt = mu_lt

    return mu_lt

def acquire(model, pool, device, B=30, M=10000, sample_num=20, sampling_index=4, score_batch_size=40):
    model.eval()
    
    origin_pos_lt = []
    pool_pred_lt = []
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            origin_pos_lt += list(pos.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)
    origin_pos_lt = np.array(origin_pos_lt)
    pool_size = pool.size()
   
    downsample_num = 5000
    sample_acq = get_single_acq(pool_pred_lt, sample_num, device, score_batch_size) 
    #backup_pool_pred_lt, backup_origin_pos_lt = pool_pred_lt, origin_pos_lt
    downsampled_pool_pred_lt, downsampled_sample_index_lt = downsample(pool_pred_lt, sample_acq, downsample_num)
    downsampled_sample_acq, downsampled_pair_acq = get_single_pair_f(downsampled_pool_pred_lt, sample_num, device, score_batch_size)

    print(f'max sample_acq: {np.max(sample_acq)}')
    downsampled_If = get_If(downsampled_sample_acq, downsampled_pair_acq)
    
    #np.savetxt('sample_acq.txt', downsampled_sample_acq)
    #np.savetxt('pair_acq.txt', downsampled_pair_acq)
    #np.savetxt('If.txt', downsampled_If)

    mu_lt = acq_driven_clustering(downsampled_sample_acq, downsampled_If, cluster_num=1000)    
    
    mu_lt = downsampled_sample_index_lt[mu_lt]
    
    print(f'final mu lt:')
    print(mu_lt)
    
    cluster2pos, pos2cluster = assign_cluster(pool_pred_lt, mu_lt, sample_acq, sample_num, device, score_batch_size)
    
    '''
    print(f'cluster2pos: {cluster2pos}')
    print(f'pos2cluster: {pos2cluster}')
    '''
    for cluster_label in cluster2pos:
        print(f'cluster_label: {cluster_label}, mu: {mu_lt[cluster_label]}, num: {len(cluster2pos[cluster_label])}')
        print(f'\t cluster pos: {cluster2pos[cluster_label]}')
    
    sel_pos_lt = []
    cluster2pos_utility = {}
    for cluster_label in cluster2pos:
        cluster2pos_utility[cluster_label] = []
        for pos in cluster2pos[cluster_label]:
            cluster2pos_utility[cluster_label].append((pos, sample_acq[pos]))
        cluster2pos_utility[cluster_label].sort(key=lambda x:x[1], reverse=True)
        #print(f'cluster: {cluster_label}')
        #print(cluster2pos_utility[cluster_label])
        i = cluster2pos_utility[cluster_label][0][0]
        pos = origin_pos_lt[i]
        sel_pos_lt.append(pos)

    print(f'select pos')
    print(sel_pos_lt)

    return sel_pos_lt
    


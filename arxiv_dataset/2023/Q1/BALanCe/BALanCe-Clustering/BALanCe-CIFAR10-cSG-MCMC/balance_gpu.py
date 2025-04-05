import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
import os
import random
from tqdm import tqdm
import models
import torch.nn.functional as F
import glob

def get_If(sample_acq, pair_acq):
    sample_num = len(sample_acq)
    If = [[0.0]*sample_num for _ in range(sample_num)]
    for i in range(sample_num):
        for j in range(sample_num):
            If[i][j] = sample_acq[i] + sample_acq[j] - pair_acq[i][j]
    return If

def downsample(pool_pred_lt1, pool_pred_lt2, sample_acq, downsample_num, coldness=1.0):
    sample_acq = np.array(sample_acq)
    log_sample_acq = np.log(sample_acq)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=sample_acq.shape)
    index_lt = np.argpartition(gumbel, -downsample_num)[-downsample_num:]

    downsample_pool_pred_lt1 = pool_pred_lt1[index_lt]
    downsample_pool_pred_lt2 = pool_pred_lt2[index_lt]
    return downsample_pool_pred_lt1, downsample_pool_pred_lt2, index_lt

def get_single_acq(pool_pred_lt1, pool_pred_lt2, hamming_dis_lt, sample_num, device, score_batch_size):
    pool_pred_lt1_pre = np.ones((pool_pred_lt1.shape[1], 1), dtype=np.float32)
    pool_pred_lt2_pre = np.ones((pool_pred_lt2.shape[1], 1), dtype=np.float32)

    pool_pred_lt1_pre_cuda = torch.from_numpy(pool_pred_lt1_pre).to(device)
    pool_pred_lt2_pre_cuda = torch.from_numpy(pool_pred_lt2_pre).to(device)
    hamming_dis_lt_cuda = torch.from_numpy(hamming_dis_lt).to(device)

    pos_lt = []
    pool_size = len(pool_pred_lt1)
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
    return acq_sample


def get_acq_pos_other(pos, pool_pred_lt1, pool_pred_lt2, hamming_dis_lt, device, sample_num, score_batch_size):
    pool_pred_lt1_pre = np.ones((pool_pred_lt1.shape[1], 1), dtype=np.float32)
    pool_pred_lt2_pre = np.ones((pool_pred_lt2.shape[1], 1), dtype=np.float32)

    pool_pred_lt1_pre_cuda = torch.from_numpy(pool_pred_lt1_pre).to(device)
    pool_pred_lt2_pre_cuda = torch.from_numpy(pool_pred_lt2_pre).to(device)

    pos_pool_pred_lt1_cuda = torch.from_numpy(pool_pred_lt1[pos]).to(device)
    pool_pred_lt1_pre_cuda = torch.matmul(pool_pred_lt1_pre_cuda[:,:,None], pos_pool_pred_lt1_cuda[:,None,:])
    tmp_size = pool_pred_lt1_pre_cuda.size()
    pool_pred_lt1_pre_cuda = pool_pred_lt1_pre_cuda.view((tmp_size[0],-1))

    pos_pool_pred_lt2_cuda = torch.from_numpy(pool_pred_lt2[pos]).to(device)
    pool_pred_lt2_pre_cuda = torch.matmul(pool_pred_lt2_pre_cuda[:,:,None], pos_pool_pred_lt2_cuda[:,None,:])
    tmp_size = pool_pred_lt2_pre_cuda.size()
    pool_pred_lt2_pre_cuda = pool_pred_lt2_pre_cuda.view((tmp_size[0],-1)) 
    
    hamming_dis_lt_cuda = torch.from_numpy(hamming_dis_lt).to(device)
    pool_size = len(pool_pred_lt1)
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
    
    return acq_sample

def get_single_pair_f(downsampled_pool_pred_lt1, downsampled_pool_pred_lt2, hamming_dis_lt, sample_num, device, score_batch_size):
    acq_sample = get_single_acq(downsampled_pool_pred_lt1, downsampled_pool_pred_lt2, hamming_dis_lt, sample_num, device, score_batch_size)
    pair_result = []
    downsample_num = len(downsampled_pool_pred_lt1)
    for pos in tqdm(range(downsample_num)):
        pos_result = get_acq_pos_other(pos, downsampled_pool_pred_lt1, downsampled_pool_pred_lt2, hamming_dis_lt, device, sample_num=sample_num, score_batch_size=score_batch_size)
        pair_result.append(pos_result)

    for i in range(len(acq_sample)):
        pair_result[i][i] = acq_sample[i]

    return acq_sample, pair_result

def assign_cluster(pool_pred_lt1, pool_pred_lt2, hamming_dis_lt, mu_lt, sample_acq, sample_num, device, score_batch_size):
    centroid2other_lt = []
    for pos in tqdm(mu_lt):
        centroid2other_lt.append(get_acq_pos_other(pos, pool_pred_lt1, pool_pred_lt2, hamming_dis_lt, device, sample_num=sample_num, score_batch_size=score_batch_size))
        for j in range(len(pool_pred_lt1)):
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

    #print('initial cluster2pos')
    #print(cluster2pos)
    for i in range(len(pool_pred_lt1)):
        if i in mu_lt:
            #print(f'\t {i} in mu_lt')
            pass
        else:
            cluster = argmax_centroid[i]
            cluster2pos[cluster].append(i)

    #print('check pos')
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
    print(f'begin acq driven clustering')
    sample_acq = np.array(sample_acq)
    If = np.array(If)
    log_sample_acq = np.log(sample_acq)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=log_sample_acq.shape)
    seed_lt = np.argpartition(gumbel, -cluster_num)[-cluster_num:]
    iteration = 0
    while True:
        iteration += 1
        print(f'\t Iteration: {iteration}')
        sub_If = If[:, seed_lt]
        print(f'\t mu_lt: {seed_lt}')
        argmax_index = np.argmax(sub_If, axis=1)
        cluster_lt = [[seed_lt[i]] for i in range(len(seed_lt))]

        for i in range(len(sample_acq)):
            if i not in seed_lt:
                cluster_lt[argmax_index[i]].append(i)

        cluster_element_val = [[] for i in range(len(cluster_lt))]
        for i in range(len(cluster_lt)):
            #if len(cluster_lt[i]) == 1:
            #    cluster_element_val[i] = [(cluster_lt[i][0], sample_acq[seed_lt[i]])]

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

        print(f'\t mu_lt:{mu_lt}')
        mu_lt = np.array(mu_lt)

        if np.all(mu_lt == seed_lt):
            break
        seed_lt = mu_lt

    return mu_lt

def acquire(model_dir, pool, hamming_pool, device, coldness=1.0, downsample_num=5000, B=10, hamming_dis_threshold=0.05, score_batch_size=20):
    assert B < downsample_num
    cluster_num=B

    net = models.ResNet18().to(device)

    dataset1 = MyDataset(pool)
    dataset2 = MyDataset(hamming_pool)
    dataloader1 = torch.utils.data.DataLoader(
            dataset1,
            batch_size=64,
            shuffle=False,
            num_workers=8
            )
    dataloader2 = torch.utils.data.DataLoader(
            dataset2,
            batch_size=64,
            shuffle=False,
            num_workers=8
            )

    origin_pos_lt = []
    pool_pred_lt = []
    hamming_pred_lt = []

    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pool_pred_lt = []
        tmp_hamming_pred_lt = []
        tmp_origin_pos_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader1):
                x, label = x.to(device), label.to(device)
                outputs, _ = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_pool_pred_lt += list(outputs.data.cpu().numpy())
                tmp_origin_pos_lt += list(pos.data.cpu().numpy())

        pool_pred_lt.append(tmp_pool_pred_lt)
        origin_pos_lt.append(tmp_origin_pos_lt)

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader2):
                x, label = x.to(device), label.to(device)
                outputs, _ = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_hamming_pred_lt += list(outputs.data.cpu().numpy())

        hamming_pred_lt.append(tmp_hamming_pred_lt)

    pool_pred_lt = np.array(pool_pred_lt)
    pool_pred_lt = np.transpose(pool_pred_lt, (1, 0, 2))

    hamming_pred_lt = np.array(hamming_pred_lt)
    hamming_pred_lt = np.transpose(hamming_pred_lt, (1, 0, 2))

    if len(origin_pos_lt) > 1:
        assert origin_pos_lt[0] == origin_pos_lt[1]

    origin_pos_lt = origin_pos_lt[0]
    sample_num = pool_pred_lt.shape[1] // 2

    pool_pred_lt1 = pool_pred_lt[:, :sample_num, :]
    pool_pred_lt2 = pool_pred_lt[:, sample_num:, :]

    hamming_pred_lt1 = hamming_pred_lt[:, :sample_num, :]
    hamming_pred_lt2 = hamming_pred_lt[:, sample_num:, :]
    
    hamming_pred_lt1 = np.argmax(hamming_pred_lt1, axis=2)
    hamming_pred_lt2 = np.argmax(hamming_pred_lt2, axis=2)

    print(f'cal pair hypothesis distance')
    print(f'hamming threshold: {hamming_dis_threshold}')
    hamming_dis_lt = [0.0 for i in range(sample_num)]
    for i in range(sample_num):
        dis = distance.hamming(hamming_pred_lt1[:,i], hamming_pred_lt2[:,i])
        print(f'dis: {dis}')
        if dis < hamming_dis_threshold:
            hamming_dis_lt[i] = 0
        else:
            hamming_dis_lt[i] = 1.0

    hamming_dis_lt = np.array(hamming_dis_lt, dtype=np.float32)
    print(f'hamming dis: {hamming_dis_lt}')
    pool_size = pool.size()
    
    print(f'Get sample acq')
    sample_acq = get_single_acq(pool_pred_lt1, pool_pred_lt2, hamming_dis_lt, sample_num, device, score_batch_size)
    print(f'Downsample: {downsample_num}') 
    downsampled_pool_pred_lt1, downsampled_pool_pred_lt2, downsampled_sample_index_lt = downsample(pool_pred_lt1, pool_pred_lt2, sample_acq, downsample_num)
    print(f'Get pair f')
    downsampled_sample_acq, downsampled_pair_acq = get_single_pair_f(downsampled_pool_pred_lt1, downsampled_pool_pred_lt2, hamming_dis_lt, sample_num,device, score_batch_size)
    print(f'Aet I_f matrix')
    downsampled_If = get_If(downsampled_sample_acq, downsampled_pair_acq)
    print(f'Acq driven clustering')
    mu_lt = acq_driven_clustering(downsampled_sample_acq, downsampled_If, cluster_num=cluster_num)
    mu_lt = downsampled_sample_index_lt[mu_lt]
    sel_pos_lt = []
    for i in range(len(mu_lt)):
        mu = mu_lt[i]
        sel_pos_lt.append(origin_pos_lt[mu])
    assert len(sel_pos_lt) == B
    return sel_pos_lt
    
    

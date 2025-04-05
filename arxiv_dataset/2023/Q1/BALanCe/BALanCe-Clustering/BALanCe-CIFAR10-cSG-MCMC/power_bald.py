import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import models
import os, glob

def acquire(model_dir, pool, device, coldness=1.0, B=30, score_batch_size=40):
    net = models.ResNet18().to(device)
    origin_pos_lt = []
    pool_pred_lt = []
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pool_pred_lt = []
        tmp_origin_pos_lt = []
        
        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)
                outputs, _ = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_pool_pred_lt += list(outputs.data.cpu().numpy())
                tmp_origin_pos_lt += list(pos.data.cpu().numpy())

        pool_pred_lt.append(tmp_pool_pred_lt)
        origin_pos_lt.append(tmp_origin_pos_lt)

    pool_pred_lt = np.array(pool_pred_lt)
    pool_pred_lt = np.transpose(pool_pred_lt, (1, 0, 2))

    if len(origin_pos_lt) > 1:
        assert origin_pos_lt[0] == origin_pos_lt[1]

    origin_pos_lt = origin_pos_lt[0]
    
    sample_num = pool_pred_lt.shape[1]

    pool_pred_lt_pre_cuda = torch.ones((sample_num, 1), device=device)
    pos_lt = []
    pool_size = len(origin_pos_lt)

    acq_sample = []
    start = 0
    while start < pool_size:
        #print(f'small b: {b}, start: {start}')
        end = start + score_batch_size
        if end > pool_size:
            end = pool_size
    
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
    
        acq_sample += list(batch_acq_sample_cuda.cpu().numpy())

        start += score_batch_size

    acq_sample = np.array(acq_sample)
    acq_sample[acq_sample <= 0] = 1e-6
    log_sample_acq = np.log(acq_sample)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=log_sample_acq.shape)
    index_lt = np.argpartition(gumbel, -B)[-B:]
    pos_lt = []
    for index in index_lt:
        pos_lt.append(origin_pos_lt[index])
    return pos_lt
 
    
   

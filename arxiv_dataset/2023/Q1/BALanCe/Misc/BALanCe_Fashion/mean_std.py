import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils

def mean_stddev_acquisition_function(logits_b_K_C):
    '''
    copy from https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0/src
    '''
    return torch_utils.mean_stddev(logits_b_K_C)

def delta_mean_std(model, pool, device, sample_num=20):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    #pool_pred_lt = []
    acq_sample = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            #label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            batch_acq_sample = mean_stddev_acquisition_function(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)

    acq_sample = np.array(acq_sample)
    pos = np.argmax(acq_sample)
    return acq_sample, pos


def delta_mean_std_batch(model, pool, device, B=10, sample_num=20):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    #pool_pred_lt = []
    acq_sample = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            #label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            #pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            batch_acq_sample = mean_stddev_acquisition_function(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)

    acq_sample = np.array(acq_sample)
    tmp_lt =  [(i, acq_sample[i]) for i in range(len(acq_sample))]
    tmp_lt.sort(key=lambda x:x[1], reverse=True)
    pos_lt = []
    for i in range(B):
        pos = tmp_lt[i][0]
        pos_lt.append(pos)
    
    return acq_sample, pos_lt



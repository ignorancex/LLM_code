import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils

def variation_ratios(logits_b_K_C):
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) - torch.exp(
        torch.max(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0]
    )

def delta_variation_ratios(model, pool, device, sample_num=20):
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
            batch_acq_sample = variation_ratios(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)

    acq_sample = np.array(acq_sample)
    pos = np.argmax(acq_sample)
    return acq_sample, pos

def delta_variation_ratios_batch(model, pool, device, B=10, sample_num=20):
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
            batch_acq_sample = variation_ratios(pred)
            batch_acq_sample = batch_acq_sample.cpu().numpy()
            acq_sample += list(batch_acq_sample)

    acq_sample = np.array(acq_sample)
    tmp_lt = [(i, acq_sample[i]) for i in range(len(acq_sample))]
    tmp_lt.sort(key=lambda x:x[1], reverse=True)
    pos_lt = []
    for i in range(B):
        pos = tmp_lt[i][0]
        pos_lt.append(pos)
    return acq_sample, pos_lt



import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os

def power_bald(model, pool, device, B=10, sample_num=20, coldness=1.0):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=6,
            shuffle=False
            )
    
    pool_pred_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            label_soft_pred = torch.nn.Softmax(dim=2)(pred)
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt)
    p_y = np.mean(pool_pred_lt, axis=1)
    entropy_p_y = entropy(p_y, axis=1)
   
    entropy_p_w_y = entropy(pool_pred_lt, axis=2)
    mean_entropy_p_w_y = np.mean(entropy_p_w_y, axis=1)

    delta_mis_sample = entropy_p_y - mean_entropy_p_w_y

    acq_sample = delta_mis_sample
    acq_sample[acq_sample <= 0] = 1e-6
    log_sample_acq = np.log(acq_sample)
    gumbel = log_sample_acq + np.random.gumbel(scale=coldness, size=log_sample_acq.shape)
    index_lt = np.argpartition(gumbel, -B)[-B:]
    return list(index_lt)

    #pos_lt = []
    #for index in index_lt:
    #    pos_lt.append(origin_pos_lt[index])
    #return pos_lt
    #pos = np.argmax(delta_mis_sample)
    #return delta_mis_sample, pos




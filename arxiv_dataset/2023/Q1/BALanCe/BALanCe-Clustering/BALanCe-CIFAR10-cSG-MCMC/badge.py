import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import os
import torch_utils
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb
import glob
import torch.nn.functional as F
import models

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def acquire(model_dir, pool, device, B=10):
    net = models.ResNet18().to(device)
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    origin_pos_lt = []
    pool_feature_lt = []
    pool_pred_lt = []

    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pool_feature_lt = []
        tmp_pool_pred_lt = []
        tmp_origin_pos_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)
                outputs, feature = net(x)
                outputs = F.softmax(outputs, dim=1)
                tmp_pool_pred_lt += list(outputs.data.cpu().numpy())
                tmp_pool_feature_lt += list(feature.data.cpu().numpy())
                tmp_origin_pos_lt += list(pos.data.cpu().numpy())

        pool_feature_lt.append(tmp_pool_feature_lt)
        pool_pred_lt.append(tmp_pool_pred_lt)
        origin_pos_lt.append(tmp_origin_pos_lt)

    pool_feature_lt = np.array(pool_feature_lt)
    pool_feature_lt = np.transpose(pool_feature_lt, (1, 0, 2))

    pool_pred_lt = np.array(pool_pred_lt)
    pool_pred_lt = np.transpose(pool_pred_lt, (1, 0, 2))

    origin_pos_lt = origin_pos_lt[0]

    embedding_lt = []
    pool_size = len(origin_pos_lt)

    start = 0
    while start < pool_size:
        end = start + 128
        if end > pool_size:
            end = pool_size

        pred = pool_pred_lt[start:end]
        pred = np.mean(pred, axis=1)

        feature = pool_feature_lt[start:end]
        feature = np.mean(feature, axis=1)
        
        nLab = pred.shape[1]
        emb_dim = feature.shape[1]

        embedding = np.zeros([end-start, emb_dim * nLab])

        max_inds = np.argmax(pred, 1)
        
        for j in range(end-start):
            for c in range(nLab):
                if c == max_inds[j]:
                    embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (1-pred[j][c])
                else:
                    embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (-1*pred[j][c])
        embedding_lt += list(embedding)
        start += 128


    #acq_sample = np.array(acq_sample)
    embedding_lt = np.array(embedding_lt)
    chosen = init_centers(embedding_lt, B) 
    sel_pos_lt = []
    for i in chosen:
        sel_pos_lt.append(origin_pos_lt[i])

    return sel_pos_lt

    '''
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            batch_size = x.size(0)
            pred = model(x, sample_num)
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            feature = activation['fc1']
            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()

            nLab = pred_exp.shape[1]
            emb_dim = feature.shape[1]
            
            embedding = np.zeros([len(pos), emb_dim * nLab])
            
            max_inds = np.argmax(pred_exp, 1)

            for j in range(len(pos)):
                for c in range(nLab):
                    if c == max_inds[j]:
                        embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (1-pred_exp[j][c])
                    else:
                        embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (-1*pred_exp[j][c])

            embedding_lt += list(embedding)

            #batch_acq_sample = batch_acq_sample.cpu().numpy()
            #acq_sample += list(batch_acq_sample)
            
            origin_pos_lt += list(pos.data.cpu().numpy())



    '''


import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical

import os
from sklearn.metrics import pairwise_distances
from scipy import stats
from copy import deepcopy

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

def delta_badge(model, pool, device, B=3, M=10000, sample_num=20, sampling_index=4, class_num=10, score_batch_size=40):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    pool_pred_lt = []
    embedding_lt = []

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            #print(input)
            activation[name] = input[0].detach()
        return hook

    model.fc2.register_forward_hook(get_activation('fc2'))

    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            pred = model(x, sample_num)
            
            pred_exp = pred.exp_()
            pred_exp = torch.mean(pred_exp, dim=1)
            pred_exp = pred_exp.data.cpu().numpy()

            batch_size = x.size(0)
            feature = activation['fc2']
            #print(f'feature size: {feature.size()}')

            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            feature = feature.data.cpu().numpy()

            pool_pred_lt += list(pred_exp)

            nLab = pred_exp.shape[1]
            emb_dim = feature.shape[1]
            embedding = np.zeros([batch_size, emb_dim * nLab])

            max_inds = np.argmax(pred_exp, 1)
            for j in range(batch_size):
                for c in range(nLab):
                    if c == max_inds[j]:
                        embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (1-pred_exp[j][c])
                    else:
                        embedding[j][emb_dim*c: emb_dim*(c+1)] = deepcopy(feature[j]) * (-1*pred_exp[j][c])

            embedding_lt += list(embedding)


    embedding_lt = np.array(embedding_lt)
    chosen = init_centers(embedding_lt, B)
    
    return None, chosen

    '''
    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)

    pool_size = pool.labels.shape[0]
    '''


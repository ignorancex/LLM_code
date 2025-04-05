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

def acquire(model, pool, device, B=10, sample_num=20):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=6
            )
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.classifier[3].register_forward_hook(get_activation('fc1'))

    origin_pos_lt = []
    acq_sample = []
    embedding_lt = []

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

    #acq_sample = np.array(acq_sample)
    embedding_lt = np.array(embedding_lt)
    chosen = init_centers(embedding_lt, B) 
    sel_pos_lt = []
    for i in chosen:
        sel_pos_lt.append(origin_pos_lt[i])

    return sel_pos_lt

    '''
    cluster2pos = pool.cluster2pos
    pos2cluster = pool.pos2cluster
    cluster_num_lt = [(cluster_label, len(cluster2pos[cluster_label])) for cluster_label in cluster2pos]
    cluster_num_lt.sort(key=lambda x:x[1])

    acquire_num_lt = []
    total_remain = 0

    for i in range(len(cluster_num_lt)):
        acquire_num_lt.append([cluster_num_lt[i][0], 0])
        total_remain += cluster_num_lt[i][1]

    assert total_remain >= B

    b = 0
    i = 0
    while b < B:
        while acquire_num_lt[i][1] >= cluster_num_lt[i][1]:
            i += 1
            i = i % len(acquire_num_lt)

        acquire_num_lt[i][1] += 1
        i += 1
        i = i % len(acquire_num_lt)
        b += 1
    
    cluster2subset_size = {}
    for i in range(len(acquire_num_lt)):
        cluster_label = acquire_num_lt[i][0]
        subset_size = acquire_num_lt[i][1]
        cluster2subset_size[cluster_label] = subset_size

    pos2utility = {origin_pos_lt[i]:acq_sample[i] for i in range(len(acq_sample))}
    cluster2pos_utility = {}
    sel_pos_lt = []
    for cluster_label in cluster2pos:
        cluster2pos_utility[cluster_label] = []
        for pos in cluster2pos[cluster_label]:
            cluster2pos_utility[cluster_label].append((pos, pos2utility[pos]))

        cluster2pos_utility[cluster_label].sort(key=lambda x:x[1], reverse=True)
        subset_size = cluster2subset_size[cluster_label]

        for i in range(subset_size):
            sel_pos_lt.append(cluster2pos_utility[cluster_label][i][0])
    
    assert len(sel_pos_lt) == B
    return sel_pos_lt
    '''


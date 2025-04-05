import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import random
from torch.distributions.categorical import Categorical
import kcenter_greedy
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import json

def cluster(model, pool, device, sample_num=50, cluster_num=100, cluster_method='HAC-assign'):
    model.eval()
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    origin_pos_lt = []
    pool_pred_lt = []
    feature_lt = []

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.classifier[3].register_forward_hook(get_activation('fc1'))

    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            batch_size = x.size()[0]
            pred = model(x, sample_num)
            label_soft_pred = pred.exp_()
            pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
            origin_pos_lt += list(pos.data.cpu().numpy())
            feature = activation['fc1']
            #print(feature.size())
            feature = feature.view((batch_size, sample_num, -1))
            feature = torch.mean(feature, dim=1)
            #print(feature.size())
            feature_lt += list(feature.data.cpu().numpy())

    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)
    pool_size = pool.size()
    feature_lt = np.array(feature_lt)
    
    if cluster_method == 'kcenter':
        kcg = kcenter_greedy.KCenterGreedy(feature_lt)
        batch_center = kcg.select_batch([], cluster_num)
        cluster_label_lt = kcg.get_labels(batch_center)
    elif cluster_method == 'kmeans++':
        kmeans = KMeans(n_clusters=cluster_num, init='k-means++', random_state=0).fit(feature_lt)
        cluster_label_lt = kmeans.labels_
    elif cluster_method == 'HAC':
        hac = AgglomerativeClustering(n_clusters=cluster_num).fit(feature_lt) 
        cluster_label_lt = hac.labels_
    elif cluster_method == 'HAC-assign':
        index_lt = [i for i in range(len(feature_lt))]
        random.shuffle(index_lt)
        sub_index_lt = index_lt[:5*cluster_num]
        sub_feature_lt = feature_lt[sub_index_lt]
        
        subset_size = 10*cluster_num

        print(f'run hac')
        hac = AgglomerativeClustering(n_clusters=cluster_num).fit(sub_feature_lt) 
        sub_cluster_label_lt = hac.labels_
        from sklearn.neighbors import NearestCentroid
        
        print(f'nearest centroid')
        clf = NearestCentroid()
        clf.fit(sub_feature_lt, sub_cluster_label_lt)
        centroid_lt = clf.centroids_

        def gen_cluster_label(inter_index_lt, return_dict):
            for index in inter_index_lt:
                vec = feature_lt[index]
                vec = np.expand_dims(vec, axis=0)
                vec = np.repeat(vec, len(centroid_lt), axis=0)
                vec = (vec - centroid_lt)**2
                dist = np.sum(vec, axis=1)
                label = np.argmin(dist)
                return_dict[index] = label

        from multiprocessing import Process, Manager
        manager = Manager()
        return_dict = manager.dict()
        proc_num = 30
        inter = int(len(feature_lt) / proc_num) + 1
        p_lt = []

        for i in range(proc_num):
            start = i*inter
            end = (i+1)*inter
            if end > len(feature_lt):
                end = len(feature_lt)
            inter_index_lt = index_lt[start:end]
            p_lt.append(Process(target=gen_cluster_label, args=(inter_index_lt, return_dict)))
            p_lt[i].start()

        for i in range(proc_num):
            p_lt[i].join()

        assert len(return_dict) == len(index_lt)
        cluster_label_lt = [0]*len(index_lt)
        for index in return_dict:
            label = return_dict[index]
            cluster_label_lt[index] = label

        mis_num = 0
        for i_index in range(5*cluster_num):
            i = index_lt[i_index]
            vec = feature_lt[i]
            vec = np.expand_dims(vec, axis=0)
            vec = np.repeat(vec, len(centroid_lt), axis=0)
            vec = (vec - centroid_lt)**2
            dist = np.sum(vec, axis=1)
            label = np.argmin(dist)
            if sub_cluster_label_lt[i_index] != label:
                mis_num += 1

        print(f'HAC-assign mis label rate: {mis_num/(subset_size)}') 

    else:
        print('no such method')
        sys.exit()

    cluster2pos = {}
    pos2cluster = {}

    for i in range(len(cluster_label_lt)):
        pos = int(origin_pos_lt[i])
        cluster_label = int(cluster_label_lt[i])
        if pos not in pos2cluster:
            pos2cluster[pos] = cluster_label
        else:
            print(f'pos already in pos2cluster')
        
        if cluster_label not in cluster2pos:
            cluster2pos[cluster_label] = [pos]
        else:
            cluster2pos[cluster_label].append(pos)

    cluster_size_lt = []
    for cluster_label in cluster2pos:
        cluster_size_lt.append(len(cluster2pos[cluster_label]))

    print(f'max cluster: {max(cluster_size_lt)}')
    print(f'min cluster: {min(cluster_size_lt)}')
    cluster_size_lt.sort()
    print(cluster_size_lt)
    pool.cluster2pos = cluster2pos
    pool.pos2cluster = pos2cluster

    with open('cluster_data/cluster2pos.json', 'w') as outfile:
        json.dump(cluster2pos, outfile)

    with open('cluster_data/pos2cluster.json', 'w') as outfile:
        json.dump(pos2cluster, outfile)
   



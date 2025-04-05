import numpy as np
from DataLoader import MyDataset
import torch
import torch.utils.data
from scipy.spatial import distance
from scipy.stats import entropy
import torch.nn.functional as F
import random
from torch.distributions.categorical import Categorical
import kcenter_greedy
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import json
from sklearn import svm
import models
import glob

cell_size=400

class Node:
    left_node=None
    right_node=None
    lin_classifier=None
    label=None

def hierachical_partition(feature_lt, original_pos_lt):
    pos2cluster = {}
    cluster2pos = {}
    index = 0

    def recursive_divide(feature_lt, original_pos_lt):
        print(f'sample num: {len(feature_lt)}')
        nonlocal index

        head = Node()
        if len(feature_lt) <= cell_size:
            cluster2pos[index] = []
            for i in range(len(original_pos_lt)):
                pos = int(original_pos_lt[i])
                pos2cluster[pos] = index
                cluster2pos[index].append(pos)
            head.label = index
            index += 1
            return head

        kmeans = KMeans(n_clusters=2, init='k-means++').fit(feature_lt)
        cluster_label_lt = kmeans.labels_
        #print(f'{type(cluster_label_lt)}')
        print(cluster_label_lt)
        print('train clf')
        clf = svm.SVC(kernel='linear')
        clf.fit(feature_lt, cluster_label_lt)
        acc = clf.score(feature_lt, cluster_label_lt)
        print(f'finish training, acc: {acc}')

        head.lin_classifier = clf
        
        left_feature_lt = feature_lt[cluster_label_lt==0]
        left_original_pos_lt = original_pos_lt[cluster_label_lt==0]
        right_feature_lt = feature_lt[cluster_label_lt==1]
        right_original_pos_lt = original_pos_lt[cluster_label_lt==1]

        head.left_node = recursive_divide(left_feature_lt, left_original_pos_lt)
        head.right_node = recursive_divide(right_feature_lt, right_original_pos_lt)

        return head

    head = recursive_divide(feature_lt, original_pos_lt)
    return head, cluster2pos, pos2cluster

def hier_clf_one(tree, feature):
    head = tree
    if head.label != None:
        return head.label
    pred = head.lin_classifier.predict(feature)
    #print(f'pred: {pred}')
    if pred[0] < 0.5:
        label = hier_clf_one(head.left_node, feature)
    else:
        label = hier_clf_one(head.right_node, feature)
    return label

def hier_clf(tree, feature_lt):
    label_lt = []
    for i in range(len(feature_lt)):
        feature = feature_lt[i:i+1]
        label = hier_clf_one(tree, feature)
        label_lt.append(label)
    return label_lt

def lin_partition(model_dir, pool, pool_val, device):
    net = models.ResNet18().to(device)
    
    dataset = MyDataset(pool)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    pool_original_pos_lt = []
    pool_pred_lt = []
    pool_feature_lt = []

    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pool_pred_lt = []
        tmp_pool_original_pos_lt = []
        tmp_pool_feature_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)
                batch_size = x.size()[0]
                pred, feature = net(x)
                pred = F.softmax(pred, dim=1)
                tmp_pool_pred_lt += list(pred.data.cpu().numpy())
                tmp_pool_original_pos_lt += list(pos.data.cpu().numpy())
                tmp_pool_feature_lt += list(feature.data.cpu().numpy())
        
        pool_pred_lt.append(tmp_pool_pred_lt)
        pool_original_pos_lt.append(tmp_pool_original_pos_lt)
        pool_feature_lt.append(tmp_pool_feature_lt)

    pool_pred_lt = np.array(pool_pred_lt, dtype=np.float32)
    pool_pred_lt = np.mean(pool_pred_lt, axis=0)

    pool_feature_lt = np.array(pool_feature_lt)
    pool_feature_lt = np.mean(pool_feature_lt, axis=0)

    pool_original_pos_lt = np.array(pool_original_pos_lt[0])
    pool_size = pool.size()
    
    dataset = MyDataset(pool_val)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8
            )
    
    val_original_pos_lt = []
    val_pred_lt = []
    val_feature_lt = []
    val_label_lt = []
 
    for path in glob.glob(f'{model_dir}/*pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_val_pred_lt = []
        tmp_val_original_pos_lt = []
        tmp_val_feature_lt = []
        tmp_val_label_lt = []

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)
                batch_size = x.size()[0]
                pred, feature = net(x)
                pred = F.softmax(pred, dim=1)
                tmp_val_pred_lt += list(pred.data.cpu().numpy())
                tmp_val_original_pos_lt += list(pos.data.cpu().numpy())
                tmp_val_feature_lt += list(feature.data.cpu().numpy())
                tmp_val_label_lt += list(label.data.cpu().numpy())
        
        val_pred_lt.append(tmp_val_pred_lt)
        val_original_pos_lt.append(tmp_val_original_pos_lt)
        val_feature_lt.append(tmp_val_feature_lt)
        val_label_lt.append(tmp_val_label_lt)

    val_pred_lt = np.array(val_pred_lt, dtype=np.float32)
    val_pred_lt = np.mean(val_pred_lt, axis=0)

    val_feature_lt = np.array(val_feature_lt)
    val_feature_lt = np.mean(val_feature_lt, axis=0)

    val_original_pos_lt = np.array(val_original_pos_lt[0])
    val_label_lt = np.array(val_label_lt[0])

    val_size = pool_val.size()
   
    print(f'val acc: {np.mean(val_label_lt==np.argmax(val_pred_lt, axis=1))}')

    tree, cluster2pos, pos2cluster  = hierachical_partition(pool_feature_lt, pool_original_pos_lt)
    
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

    label_lt = hier_clf(tree, val_feature_lt)    
    #print(label_lt)
    
    val_cluster2index = {}
    for i in range(len(label_lt)):
        label = label_lt[i]
        if label not in val_cluster2index:
            val_cluster2index[label] = [i]
        else:
            val_cluster2index[label].append(i)

    val_cluster2acc = {}
    for label in val_cluster2index:
        tmp_index =  val_cluster2index[label]
        tmp_pred_lt = val_pred_lt[tmp_index]
        tmp_label_lt = val_label_lt[tmp_index]
        
        tmp_pred_lt = np.argmax(tmp_pred_lt, axis=1)
        acc = np.mean(tmp_pred_lt == tmp_label_lt)
        val_cluster2acc[label] = acc
        print(f'cluster: {label}, cluster_size: {len(val_cluster2index[label])}, acc: {acc}')
    
    print(val_cluster2acc)

    val_cluster2pos = {}
    val_pos2cluster = {}

    for label in val_cluster2index:
        val_cluster2pos[label] = []
        for index in val_cluster2index[label]:
            pos = int(val_original_pos_lt[index])
            val_cluster2pos[label].append(pos)
            val_pos2cluster[pos] = label

    count = 0 
    acc = 0.
    for label in val_cluster2acc:
        acc += val_cluster2acc[label]*len(val_cluster2index[label])
        count += len(val_cluster2index[label])
    print(f'verify val acc: {acc/count}')

    pool.cluster2acc = val_cluster2acc
    pool_val.cluster2pos = val_cluster2pos
    pool_val.pos2cluster = val_pos2cluster

    with open('cluster_data/val_cluster2acc.json', 'w') as outfile:
        json.dump(val_cluster2acc, outfile)

    with open('cluster_data/val_cluster2pos.json', 'w') as outfile:
        json.dump(val_cluster2pos, outfile)

    with open('cluster_data/val_pos2cluster.json', 'w') as outfile:
        json.dump(val_pos2cluster, outfile)


    '''
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
   
    '''


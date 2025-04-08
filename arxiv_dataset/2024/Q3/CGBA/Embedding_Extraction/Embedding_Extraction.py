#!/usr/bin/env python
# coding: utf-8

import os
gpu_avail = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_avail}"


from datasets import load_dataset, Dataset, load_from_disk
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import pickle
path = '../COVID_pub'
name = 'COVID'


Claims_total = dict()
for split in ['train']:
    with open(os.path.join(path, f'{name}_{split}_Claims.json'), 'r') as file:
        claims = json.load(file)
        Claims_total[split] = claims


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


sbert = SentenceTransformer('all-MiniLM-L12-v2', device=device)

claims_train= [item for sublist in Claims_total['train'].values() for item in sublist]

embeddings_train = sbert.encode(claims_train, batch_size=4096, show_progress_bar=True, device=device)

embeddings = embeddings_train

with open('SBERT_COVID_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

for val in [11]:
    dbscan = DBSCAN(eps=val, min_samples=3)
    clusters = dbscan.fit_predict(embeddings_scaled)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print("-"*30)
    print(val)
    print(f'Number of clusters: {n_clusters}')
    print(f'Number of points classified as noise: {n_noise}')
    
    cluster_counts = Counter(clusters)

    # Print the number of points in each cluster
    n_points = []
    for cluster_id, num_points in cluster_counts.items():
        n_points.append(num_points)
    n_points_sorted = sorted(n_points, reverse=True)

    print(n_points_sorted[:20])

        # Evaluate the result of clustering
    if n_clusters > 1:
        silhouette_avg = silhouette_score(embeddings_scaled, clusters)
        print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
    else:
        print("Not enough clusters to calculate the silhouette score.")

Sentences_total = []
for split in ['train']:
    with open(os.path.join(path, f'{name}_{split}_Sentences.json'), 'r') as file:
        sentences = json.load(file)
        Sentences_total.extend(sentences)

dataset = load_from_disk(os.path.join(path))
sen2label = dict()
for sen, label in zip(dataset['train']['sentences'], dataset['train']['label']):
    sen2label[sen] = label


for cluster, num in Counter(clusters).most_common(30):
    if cluster == -1: continue
    print('-'*40)
    print(f'Cluster:{cluster} / Data points:{num}')
    # Predefined cluster
    predefined_cluster = cluster

    # Get indexes of elements in clusters that are equal to the predefined cluster
    indexes = np.where(clusters == predefined_cluster)[0]

    matched_sentences = [Sentences_total[i] for i in indexes]
    
    print(f'Matched Sentences:{len(matched_sentences)} / Unique Matched Sentences:{len(set(matched_sentences))}')

    cnt_r, cnt_f = 0, 0
    
    for sen in set(matched_sentences):
        if sen2label[sen] == 'real': cnt_r +=1
        else: cnt_f += 1

    print(f'Real:{cnt_r} / Fake:{cnt_f}')


only_fake = dict() ## key is cluster number and the value is datapoints in it
only_fake_unique = dict()
only_real = dict() 
only_real_unique = dict()
for cluster, num in tqdm(Counter(clusters).most_common()):
    if cluster == -1: continue
    
    # Predefined cluster
    predefined_cluster = cluster

    # Get indexes of elements in clusters that are equal to the predefined cluster
    indexes = np.where(clusters == predefined_cluster)[0]

    matched_sentences = [Sentences_total[i] for i in indexes]

    cnt_r, cnt_f = 0, 0
    
    for sen in set(matched_sentences):
        if sen2label[sen] == 'real': cnt_r +=1
        else: cnt_f += 1

    if cnt_r == 0 and cnt_f > 0:
        only_fake[cluster] = num
        only_fake_unique[cluster] = len(set(matched_sentences))

    if cnt_f == 0 and cnt_r > 0:
        only_real[cluster] = num
        only_real_unique[cluster] = len(set(matched_sentences))

new_clusters = []
for d in tqdm(clusters):
    if d == -1 : 
        new_clusters.append(-1)
        continue
    if only_fake.get(d) == None and only_real.get(d) == None: ## Combined cluster
        new_clusters.append(-1)
        continue
    new_clusters.append(d)

### filtering clusters have >= 10 unique sentences
cluster_fake = dict()
for k, v in only_fake_unique.items():
    if v >= 10: cluster_fake[k] = v
cluster_real = dict()
for k, v in only_real_unique.items():
    if v >= 10: cluster_real[k] = v 

fake_clusters = [str(k) for k in cluster_fake.keys()]
real_clusters = [str(k) for k in cluster_real.keys()]

with open(os.path.join(path, 'real_clusters.json'), 'w') as file:
    json.dump(real_clusters,file)
with open(os.path.join(path, 'fake_clusters.json'), 'w') as file:
    json.dump(fake_clusters,file)

cluster_fin = dict()
cluster_fin.update(cluster_fake)
cluster_fin.update(cluster_real)

random_seed = 123
torch.manual_seed(random_seed)
random.seed(random_seed)


cluster_w_sentences = dict()
for cluster in tqdm(cluster_fin.keys()):
    indexes = np.where(clusters == cluster)[0]
    matched_sentences = list(set([Sentences_total[i] for i in indexes]))
    cluster_w_sentences[cluster] = matched_sentences    

cluster_w_test_sentences = dict()
for cluster, sens in cluster_w_sentences.items():
    test_sens = random.sample(sens, int(0.2*len(sens)))
    cluster_w_test_sentences[cluster] = test_sens

## Not included sentences in contrastive learning becuase they are used for test
deleted_sens = [] 
for c, sens in cluster_w_test_sentences.items():
    for sen in sens:
        deleted_sens.append(sen)

## Delete test sentences for new_clusters (set -1)
new_clusters_ = new_clusters.copy()
cnt = 0
for ind, (sen, cluster) in tqdm(enumerate(zip(Sentences_total, new_clusters_))):
    if sen in deleted_sens:
        new_clusters[ind] = -1
        cnt +=1

new_clusters = [str(v) for v in new_clusters]
with open(os.path.join(path, 'New_clusters.json'), 'w') as file:
    json.dump(new_clusters,file)
cluster_w_test_sentences = {str(k):v for k,v in cluster_w_test_sentences.items()}
with open(os.path.join(path, 'Cluster_w_test_sentences.json'), 'w') as file:
    json.dump(cluster_w_test_sentences,file)

cluster_w_tv_sentences = dict()
for (c1, sens1), (c2, sens2) in zip(cluster_w_test_sentences.items(), cluster_w_sentences.items()):
    train_valid_sens = [sen for sen in sens2 if sen not in sens1]
    if not (len(train_valid_sens)==len(sens2)-len(sens1)): print("DIFF")
    cluster_w_tv_sentences[c1] = train_valid_sens
with open(os.path.join(path, 'Cluster_w_train_valid_sentences.json'), 'w') as file:
    json.dump(cluster_w_tv_sentences,file)

sen_for_contrastive, cluster_for_contrastive, claims_for_contrastive = [], [], []
for sen, cluster, claim in zip(Sentences_total, new_clusters, claims_train):
    if cluster != "-1":
        sen_for_contrastive.append(sen)
        cluster_for_contrastive.append(cluster)
        claims_for_contrastive.append(claim)

dict_c = dict()
for c in cluster_for_contrastive:
    if dict_c.get(c) == None: dict_c[c] = 1
    else: dict_c[c] += 1
deleted_clusters = []
for c, v in dict_c.items():
    if v < 3: deleted_clusters.append(c) 

sentences, clusters, claims = [], [], []
for sen, cluster, claim in zip(sen_for_contrastive, cluster_for_contrastive, claims_for_contrastive):
    if cluster not in deleted_clusters:
        sentences.append(sen)
        clusters.append(cluster)
        claims.append(claim)

with open(os.path.join(path, 'Sentences_for_contrastive.json'), 'w') as file:
    json.dump(sentences,file)
with open(os.path.join(path, 'Clusters_for_contrastive.json'), 'w') as file:
    json.dump(clusters,file)
with open(os.path.join(path, 'Claims_for_contrastive.json'), 'w') as file:
    json.dump(claims,file)

import torch
import numpy as np
import random
from sklearn.cluster import KMeans
import Pool
from Buffer import Buffer

val_dataset_size = 180000
hamming_pool_size = 180000

pool = Pool.Pool('../processed/training_bymerge.pt')

buff_val = Buffer()
buff_train = Buffer()
buff_hamming = Buffer()

print( 'sep buff_vall dataset')
feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
buff_val.add_rows(feat_lt, label_lt)

'''
feat_lt, label_lt = pool.random_query_balanced_labels(K=2)
buff_train.add_rows(feat_lt, label_lt)
'''

print('sep buff_hamming dataset')
feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
buff_hamming.add_rows(feat_lt, label_lt)

training_set = (torch.from_numpy(pool.features), torch.from_numpy(pool.labels))
with open('training.pt', 'wb') as f:
    torch.save(training_set, f)

'''
start_set = (torch.from_numpy(buff_train.features), torch.from_numpy(buff_train.labels))
with open('start.pt', 'wb') as f:
    torch.save(start_set, f)
'''

val_set = (torch.from_numpy(buff_val.features), torch.from_numpy(buff_val.labels))
with open('val.pt', 'wb') as f:
    torch.save(val_set, f)
 
hamming_set = (torch.from_numpy(buff_hamming.features), torch.from_numpy(buff_hamming.labels))
with open('hamming.pt', 'wb') as f:
    torch.save(hamming_set, f)


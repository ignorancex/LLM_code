import torch
import numpy as np
import random
from sklearn.cluster import KMeans
import Pool
from Buffer import Buffer

val_dataset_size = 2000
#hamming_pool_size = 10000

pool = Pool.Pool('Dataset/train/X_train.txt', 'Dataset/train/y_train.txt')

buff_val = Buffer()
#buff_test = Buffer()
buff_train = Buffer()
buff_hamming = Buffer()

feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
buff_val.add_rows(feat_lt, label_lt)

#feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
#buff_test.add_rows(feat_lt, label_lt)

feat_lt, label_lt = pool.random_query_balanced_labels(K=2)
#feat_lt, label_lt = pool.random_query_labels(10)
buff_train.add_rows(feat_lt, label_lt)

#feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
#buff_hamming.add_rows(feat_lt, label_lt)

training_set = (torch.from_numpy(pool.features), torch.from_numpy(pool.labels))

with open('training.pt', 'wb') as f:
    torch.save(training_set, f)

start_set = (torch.from_numpy(buff_train.features), torch.from_numpy(buff_train.labels))
with open('start.pt', 'wb') as f:
    torch.save(start_set, f)

val_set = (torch.from_numpy(buff_val.features), torch.from_numpy(buff_val.labels))
with open('val.pt', 'wb') as f:
    torch.save(val_set, f)

#test_set = (torch.from_numpy(buff_test.features), torch.from_numpy(buff_test.labels))
#with open('test.pt', 'wb') as f:
#    torch.save(test_set, f)
 
hamming_set = (torch.from_numpy(pool.features), torch.from_numpy(pool.labels))
with open('hamming.pt', 'wb') as f:
    torch.save(hamming_set, f)
 
test_pool = Pool.Pool('Dataset/test/X_test.txt', 'Dataset/test/y_test.txt')
test_set = (torch.from_numpy(test_pool.features), torch.from_numpy(test_pool.labels))
with open('test.pt', 'wb') as f:
    torch.save(test_set, f)
 

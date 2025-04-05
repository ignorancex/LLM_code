import torch
import numpy as np
import random
from sklearn.cluster import KMeans
import Pool
from Buffer import Buffer

pool = Pool.Pool('./training.pt')
pool.repeat(3)

training_set = (torch.from_numpy(pool.features), torch.from_numpy(pool.labels))
with open('training_repeat.pt', 'wb') as f:
    torch.save(training_set, f)



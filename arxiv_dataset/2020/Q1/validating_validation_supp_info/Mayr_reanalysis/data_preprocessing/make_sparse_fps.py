'''
@author: Matthew C. Robinson
@email: matthew67robinson@gmail.com
'''

import math
import itertools
import numpy as np
import scipy
import scipy.io
import scipy.sparse

import pickle
import os
import sys
import rdkit
import rdkit.Chem
from rdkit.Chem import AllChem

with open('../data_for_replication/fp_data.pckl', 'rb') as f:
    fp_arr = pickle.load(f)

sparse_mat_list = []
for i in range(0, fp_arr.shape[0], 1000):
    if i % 10 000 == 0:
        print(i)
    np_fps = np.vstack([np.array(x) if x is not None else np.full(1024,-1) for x in fp_arr[i:(i+1000)]])
    sparse_mat_list.append(scipy.sparse.csr_matrix(np_fps))

sparse_fps_arr = scipy.sparse.vstack(tuple(sparse_mat_list)) # shape is 456331 x 1024

with open('../data_for_replication/sparse_fps_arr.pckl','wb') as f:
    pickle.dump(sparse_fps_arr,f)

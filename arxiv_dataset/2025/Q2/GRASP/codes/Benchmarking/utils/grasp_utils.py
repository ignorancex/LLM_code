import numpy as np
from scipy import sparse
import dgl
import torch

def build_graph(feats):
    mag_0 = list(feats['features'].keys())[0]
    n = len(feats['features'][mag_0]) #num_feats #:(
   
    n_mag = len(feats['features'])
    A = np.zeros((n_mag*n,n_mag*n), dtype=np.float16)
    #print("n: ", n, n_mag)
    for block_row in range(n_mag):
        for block_col in range(n_mag):
            if block_row == block_col:
                A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.ones((n,n), dtype=np.float16)
            if abs(block_row-block_col)==1:
                A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.diag(np.ones((1,n), dtype=np.float16).reshape(n,))
    
    sA = sparse.csr_matrix(A)
    g = dgl.from_scipy(sA)
    #g = g_.int()
    #print(g.idtype)
    tensor_list = list(feats['features'].values())
    coords_list = list(feats['coords'].values())
    #print(tensor_list)
    x = torch.cat(tensor_list, dim=0)
    c = torch.cat(coords_list, dim=0)
    if len(x.shape) > 2:
        g.ndata['x'] = x[:,12,:]
        g.ndata['c'] = c[:,12,:]
    else:
        g.ndata['x'] = x
        g.ndata['c'] = c
    
    return g
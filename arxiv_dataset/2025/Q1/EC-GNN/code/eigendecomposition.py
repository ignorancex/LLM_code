import datasets
import torch
import time
import sys
from torch_geometric.utils import add_self_loops,to_undirected,to_dense_adj,is_undirected
dataset = ['pubmed','texas','cora', 'citeseer', 'computers', 'squirrel', 'photo', 'chameleon', 'film',  'cornell']


for d in dataset:
    s = time.time()
    
    eigenvalues_path = '/data/eigenvalues/'+d
    eigenvectors_path = '/data/eigenvectors/'+d

    baseG = datasets.load_dataset(d, 'dense').to('cuda:0')

    print(d,baseG.num_nodes,baseG.edge_index.size())

    A =  to_dense_adj(baseG.edge_index).squeeze()
    n = baseG.num_nodes

    degree = torch.diag(A.sum(-1)**(-0.5))
    degree[torch.isinf(degree)] = 0.
    A_sym = degree.mm(A.mm(degree))

    e,U=torch.linalg.eigh(A_sym,UPLO='U')
    print(d,time.time()-s)

    torch.save(U,eigenvectors_path)
    torch.save(e,eigenvalues_path)


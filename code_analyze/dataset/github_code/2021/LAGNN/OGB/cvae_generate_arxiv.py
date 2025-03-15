import argparse
import numpy as np
import random
import scipy.sparse as sp
import cvae_pretrain_small

import torch

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')

parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=1e-5)
parser.add_argument("--total_iterations", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')


args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
adj_scipy = sp.csr_matrix(data.adj_t.to_scipy())
data = data.to(device)

#Pretrain
cvae_model = cvae_pretrain_small.generated_generator(args, device, adj_scipy, data.x)
torch.save(cvae_model, "model/arxiv.pkl")

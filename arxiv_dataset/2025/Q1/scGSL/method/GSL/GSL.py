import time
import argparse
import numpy as np
import torch
from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test
import pandas as pd
from scipy.sparse import csr_matrix

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='graph_data',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
#parser.add_argument('--attack', type=str, default='meta',
parser.add_argument('--attack', type=str, default='no',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=10, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0.1, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)


adj_df_path = r'graph.csv'
adj_df = pd.read_csv(adj_df_path, index_col=0, header=0)
adj= csr_matrix(adj_df.values)
features_path = r'feature.csv'
features_df = pd.read_csv(features_path, index_col=0, header=0).T
features= csr_matrix(features_df.values)
labels_path = r'label.csv'
labels_df = pd.read_csv(labels_path, header=0)
labels = labels_df.iloc[:, 0].copy()
nclass = labels.max().item() + 1
seed = 15
idx_train, idx_val, idx_test = get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.5, stratify=labels, seed=seed)

if args.attack == 'no':
    perturbed_adj = adj

if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    attacker.attack(adj, n_perturbations, type='add')
    perturbed_adj = attacker.modified_adj

if args.attack == 'meta' or args.attack == 'nettack':
    perturbed_data = PrePtbDataset(root='./tmp/',
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    if args.attack == 'nettack':
        idx_test = perturbed_data.target_nodes

np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device)

if args.only_gcn:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, verbose=True, train_iters=args.epochs)
    model.test(idx_test)
else:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
    prognn = ProGNN(model, args, device)
    print(prognn)
    prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)
    adjfin, acc_test = prognn.test(features, labels, idx_test)
    data_cpu = adjfin.cpu().numpy()
    tensor_df = pd.DataFrame(data=data_cpu, index=adj_df.index, columns=adj_df.columns)
    output_path = adj_df_path.rsplit('.csv', 1)[0] + '_adjfin_graph2.csv'
    tensor_df.to_csv(output_path)
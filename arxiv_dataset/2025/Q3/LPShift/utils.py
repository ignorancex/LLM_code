import torch
import numpy as np
import random
import logging, sys
import math
import logging.config 
import networkx as nx
import scipy.sparse as ssp

from scipy.stats import rankdata
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.utils import (add_self_loops, negative_sampling, degree, to_undirected)

def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())

    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index

def SP(A, edge_index, remove=True):
    
    scores = []
    G = nx.from_scipy_sparse_array(A)
    print(len(G.edges()))
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    
        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores), edge_index

def RA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def PA(A, edge_index, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    
    G = nx.from_scipy_sparse_array(A)
    G_degree = np.array(G.degree(np.array(G.nodes())))
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        temp_tup = list(zip(list(src), list(dst)))
        cur_scores = G_degree[src][:, 1] * G_degree[dst][:, 1]
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
   
      # any source is fine
    pos_edge = split_edge['train']['edge'].t()
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list

def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


## HeaRT Sampling
def prep_data(data, edge_split):
    """
    Various prep
    """
    data.adj_t = data.adj_t.coalesce().bool().float()
    data.adj_t = data.adj_t.to_symmetric()

    train_edge_index = to_undirected(edge_split['train']['edge'].t())

    val_edge_index = to_undirected(edge_split['valid']['edge'].t())
    full_edge_index = torch.cat([train_edge_index, val_edge_index], dim=-1)

    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
    train_edge_weight = torch.ones([train_edge_index.size(1), 1], dtype=torch.float)
    full_edge_weight = torch.cat([train_edge_weight, val_edge_weight], 0).view(-1)

    data.full_edge_index = full_edge_index
    data.full_edge_weight = full_edge_weight
    data.full_adj = SparseTensor.from_edge_index(full_edge_index, full_edge_weight, [data.num_nodes, data.num_nodes])
    data.full_adj = data.full_adj.to_symmetric()

    return data


def calc_CN(data, use_val=False):
    """
    Calc CNs for all node pairs
    """
    if use_val:
        adj = data.full_adj_t
    else:
        adj = data.adj_t

    cn_scores = adj @ adj

    return cn_scores

def calc_PA(data, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    
    G_degree = degree(data.edge_index[0], data.num_nodes)

    return G_degree

def rank_score_matrix(row):
    """
    Rank from largest->smallest
    """
    num_greater_zero = (row > 0).sum().item()

    # Ignore 0s and -1s in ranking
    # Note: default is smallest-> largest so reverse
    if num_greater_zero > 0:
        ranks_row = rankdata(row[row > 0], method='min')
        ranks_row = ranks_row.max() - ranks_row + 1
        max_rank = ranks_row.max()
    else:
        ranks_row = []
        max_rank = 0

    # Overwrite row with ranks
    # Also overwrite 0s with max+1 and -1s with max+2
    row[row > 0] = ranks_row
    row[row == 0] = max_rank + 1
    row[row < 0] = max_rank + 2

    return row

def rank_and_merge_node(node_scores, true_pos_mask, data, args):
    """
    Do so for a single node
    """
    k = args.num_samples // 2 

    # Nodes that are 0 for all scores. Needed later when selecting top K
    zero_nodes_score_mask = (node_scores == 0).numpy()

    # Individual ranks
    node_ranks = rank_score_matrix(node_scores.numpy())

    # If enough non-zero scores we use just take top-k
    # Otherwise we have to randomly select from 0 scores        
    max_greater_zero = data['num_nodes'] - zero_nodes_score_mask.sum().item() - true_pos_mask.sum().item()

    # NOTE: Negate when using torch.topk since 1=highest
    if max_greater_zero >= k:
        node_topk = torch.topk(torch.from_numpy(-node_ranks), k).indices
        node_topk = node_topk.numpy()
    elif max_greater_zero <= 0:
        # All scores are either true_pos or 0
        # We just sample from 0s here
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_topk = np.random.choice(node_zero_score_ids, k)
    else:
        # First just take whatever non-zeros there are
        node_greater_zero = torch.topk(torch.from_numpy(-node_ranks), max_greater_zero).indices
        node_greater_zero = node_greater_zero.numpy()

        # Then choose the rest randomly from 0 scores
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_zero_rand = np.random.choice(node_zero_score_ids, k-max_greater_zero)
        node_topk = np.concatenate((node_greater_zero, node_zero_rand))

    return node_topk.reshape(-1, 1)

def rank_and_merge_edges(edges, cn_scores, pa_scores, data, train_nodes, args, test=False):
    """
    For each edge we get the rank for the types of scores for each node and merge them together to one rank

    Using that we get the nodes with the top k ranks
    """
    all_topk_edges = []
    k = args.num_samples // 2 

    # Used to determine positive samples to filter
    # For testing we also include val samples in addition to train
    if test:
        adj = data.full_adj
    else:
        adj = data.adj_t

    if args.metric.upper() == "SP":
        edge_index, edge_weight = data.full_edge_index, data.full_edge_weight
        A_ssp = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes))
        G = nx.from_scipy_sparse_array(A_ssp)    

    ### Get nodes not in train
    all_nodes = set(list(range(data.num_nodes)))
    nodes_not_in_train = torch.Tensor(list(all_nodes - train_nodes)).long()
    
    for edge in tqdm(edges, "Ranking Scores"):
        source, target = edge[0].item(), edge[1].item()

        source_adj = adj[source].to_dense().squeeze(0).bool()
        target_adj = adj[target].to_dense().squeeze(0).bool()

        if args.metric.upper() == "CN":
            source_scores = cn_scores[source].to_dense().squeeze(0)
            target_scores = cn_scores[target].to_dense().squeeze(0)
        elif args.metric.upper() == "PA":
            source_scores = target_scores = pa_scores
        else:
            raise NotImplementedError(f"{arg.metric.upper()} is not implemented!")

        source_true_pos_mask = source_adj
        target_true_pos_mask = target_adj

        # Don't remove true positive
        # So just set all to 0
        # if args.keep_train_val:
        #     source_true_pos_mask = torch.zeros_like(source_true_pos_mask)
        #     target_true_pos_mask = torch.zeros_like(target_true_pos_mask)

        # Mask nodes not in train
        source_true_pos_mask[nodes_not_in_train] = 1
        target_true_pos_mask[nodes_not_in_train] = 1

        # Include masking for self-loops
        source_true_pos_mask[source], source_true_pos_mask[target] = 1, 1
        target_true_pos_mask[target], target_true_pos_mask[source] = 1, 1

        # Filter samples by setting to -1
        source_scores[source_true_pos_mask], source_scores[source_true_pos_mask] = -1, -1 

        source_topk_nodes = rank_and_merge_node(source_scores, source_true_pos_mask, data, args)
        source_topk_edges = np.concatenate((np.repeat(source, k).reshape(-1, 1), source_topk_nodes), axis=-1)

        target_topk_nodes = rank_and_merge_node(target_scores, target_true_pos_mask, data, args)
        target_topk_edges = np.concatenate((target_topk_nodes, np.repeat(target, k).reshape(-1, 1)), axis=-1)
        
        edge_samples = np.concatenate((source_topk_edges, target_topk_edges))
        all_topk_edges.append(edge_samples)

    return np.stack(all_topk_edges)


def calc_all_heuristics(args, data, split_edge, dataset_name):
    """
    Calc and store top-k negative samples for each sample
    """
    print("Prepping data...")
    data = prep_data(data, split_edge)

    # Get unique nodes in train
    train_nodes = set(split_edge['train']['edge'].flatten().tolist())

    print("Compute CNs...")
    cn_scores = calc_CN(data)
    print("Compute PA...")
    pa_scores = calc_PA(data)

    print("\n>>> Valid")
    val_neg_samples = rank_and_merge_edges(split_edge['valid']['edge'], cn_scores, pa_scores, data, train_nodes, args)
    with open(f"dataset/{dataset_name}Dataset/heart_valid_samples.npy", "wb") as f:
        np.save(f, val_neg_samples)

    print("\n>>> Test")
    test_neg_samples = rank_and_merge_edges(split_edge['test']['edge'], cn_scores, pa_scores, data, train_nodes, args, test=True)
    with open(f"dataset/{dataset_name}Dataset/heart_test_samples.npy", "wb") as f:
        np.save(f, test_neg_samples)
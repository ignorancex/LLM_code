import torch
import os
import numpy as np
import argparse
import random as rand

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, coalesce, is_undirected
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix

from utils import CN, PA, SP, init_seed, calc_all_heuristics

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def init_graph(dataset_name):

    if "ogbl" in dataset_name:
        temp_dname = dataset_name.split("_")
        dataset = PygLinkPropPredDataset(name=temp_dname[0])
        split_edge = dataset.get_edge_split()
        graph = dataset[0]
        ogb_nfeat = graph.x
        full_num_nodes = graph.num_nodes
        #Make EI symmetric before converting
        edge_index = to_undirected(graph.edge_index).numpy()
        edge_we = [1 for i in range(len(edge_index[0]))]
        A = coo_matrix((edge_we, (edge_index[0], edge_index[1])), shape=(full_num_nodes, full_num_nodes))
        A = A.tocsr()
    else:
        raise AssertionError("Functionality only supports ogbl datasets")

    print("EDGE INDEX: ", edge_index)    
    print("SPLIT EDGE: ", split_edge)   
    print("Total Nodes in Original Graph Object: ", full_num_nodes)
    print("Current num of non-zero elements in Adj Matrix: ", A.nnz)

    graph_dict = {"adj_matrix": A, "edge_index" : edge_index, "num_nodes" : full_num_nodes, 
                  "node_feat": ogb_nfeat, "split_edge" : split_edge}

    return graph_dict

def heur_split(graph_dict, args):
    A = graph_dict['adj_matrix']
    rows, cols = A.nonzero()
    A[rows, cols] = A[cols, rows]
    edge_index = torch.tensor(graph_dict['edge_index'])

    if args.inverse:
        print("Inverse")

    train_edges, valid_edges, test_edges = [], [], []

    assert edge_index.shape[0] == 2, "Edge Index in Heur Split not correct shape, should start with 2"
    if args.split_type == 'pa':
        heur_scores, _ = PA(A, edge_index)
    if args.split_type == 'cn':
        heur_scores, _ = CN(A, edge_index)
    if args.split_type == 'sp':
        heur_scores, _ = SP(A, edge_index)

    if args.debug_heur:
        print("Heuristic Scores Shape")
        print(heur_scores.size())
        print("Heuristic Scores")
        print(heur_scores)
        print("Heuristic Scores Unique")
        print(torch.unique(heur_scores, return_counts=True))
        print("Heuristic Scores Shape - Train")
        train_tensor = torch.where(heur_scores < args.valid_rat, 1, 0)
        print(int(torch.count_nonzero(train_tensor)))
        print("Heuristic Scores Shape - Valid")
        valid_tensor = torch.where((heur_scores < args.test_rat) & (heur_scores >= args.valid_rat), 1, 0)
        print(int(torch.count_nonzero(valid_tensor)))
        print("Heuristic Scores Shape - Test")
        test_tensor = torch.where(heur_scores >= args.test_rat, 1, 0)
        print(int(torch.count_nonzero(test_tensor)))
    #total_heur_score = float(torch.sum(heur_scores))

    edge_index = edge_index.t() # Transpose again to put edges in easily-parsed manner
    assert edge_index.shape[1] == 2, "Edge Index in Heur Split not correct shape, should end with 2"
    for i, e in tqdm(enumerate(edge_index), "Splitting", total=len(edge_index)):
        edge_tup = (int(e[0]), int(e[1]))
        cur_heur_score = float(heur_scores[i])
        #if args.split_type == 'pa': cur_heur_score = (cur_heur_score / total_heur_score) * edge_index.shape[0]

        if cur_heur_score < args.valid_rat:
            if args.inverse:
                test_edges.append(edge_tup)
            else:
                train_edges.append(edge_tup)
        elif cur_heur_score >= args.valid_rat and cur_heur_score < args.test_rat:
            valid_edges.append(edge_tup)
        elif cur_heur_score >= args.test_rat:
            if args.inverse:
                train_edges.append(edge_tup)
            else:
                test_edges.append(edge_tup)

        # else:
        #     print("Not assigned!")
        #     print(cur_heur_score)
                
    if args.debug_heur:
        print("Total Edges, Train, Valid, Test b4 filtering")
        print(len(train_edges))
        print(len(valid_edges))
        print(len(test_edges))
        print("Total train+valid+test:", len(train_edges) + len(valid_edges) + len(test_edges))
    
    train_edges = list(set(train_edges))
    valid_edges = list(set(valid_edges))
    test_edges = list(set(test_edges))

    if len(test_edges) > 100000: test_edges = rand.sample(test_edges, k=100000)
    if len(valid_edges) > 100000: valid_edges = rand.sample(valid_edges, k=100000)

    if args.debug_heur:
        print("Total Edges, Train, Valid, Test after filtering")
        print(len(train_edges))
        print(len(valid_edges))
        print(len(test_edges))
        print("Total train+valid+test:", len(train_edges) + len(valid_edges) + len(test_edges))

    return train_edges, valid_edges, test_edges

def filter_tuples(list1, training_node_set):
    fil_l = []
    
    for tuple_ in list1:
        if tuple_[0] in training_node_set and tuple_[1] in training_node_set:
            fil_l.append(tuple_)
    
    return fil_l

def fill_data_feats(graph_dict):
    edge_index = graph_dict['edge_index']
    edge_index = to_undirected(edge_index.long())
    assert is_undirected(edge_index), "Edge Index is not undirected in fill_data_feats, are the values formatted to long?"
    assert edge_index.shape[0] == 2, "Edge Index in Fill Data not correct shape, should start with 2"
    max_index = max(edge_index[0].max(), edge_index[1].max()).item()
    assert max_index < graph_dict['num_nodes'], f"Max node index ({max_index}) exceeds num_nodes ({graph_dict['num_nodes']})"
    assert (edge_index >= graph_dict['num_nodes']).nonzero().numel() == 0, f"Invalid node indices found, likely nodes present in original and not in the split."

    adj_t = SparseTensor.from_edge_index(edge_index.long(), torch.ones(edge_index.shape[1]), [graph_dict['num_nodes'], graph_dict['num_nodes']]) # Build adjacency matrix of same size as original graph to preserve edge indices
    adj_t = adj_t.to_symmetric() # Force undirected graph
    adj_t = adj_t.coalesce().bool().float()

    data = Data(edge_index=graph_dict['edge_index'], x=graph_dict['node_feat'], adj_t=adj_t, num_nodes=graph_dict['num_nodes'])
    return data

def build_synth(dataset, args):
    graph_dict = init_graph(dataset)

    assert args.split_type == 'cn' or args.split_type == 'pa' or args.split_type == 'sp', "Invalid Split Type, must be CN, PA, SP"
    train_edges, valid_edges, test_edges = heur_split(graph_dict, args)

    print("Heur Split Train/Valid/Test:", len(train_edges), 
          len(valid_edges), 
          len(test_edges))

    valid_edges_set, test_edges_set = set(valid_edges), set(test_edges)
   
    for i, e in enumerate(train_edges):
        if e in valid_edges_set or e in test_edges_set:
            raise Exception("Train Edges have overlapping edges in Valid and Test, likely issue with split")
    
    training_set = {element for tuple_ in train_edges for element in tuple_}
    valid_edges = filter_tuples(valid_edges, training_set)
    test_edges = filter_tuples(test_edges, training_set)

    filt_ei = []
    assert graph_dict['edge_index'].T.shape[1] == 2, "Edge Index within build_synth not in proper format, first index element after transpose should be 2"
    for i in graph_dict['edge_index'].T:
        i = tuple(i.tolist())
        i_inv = (i[1], i[0])
        if i in test_edges_set or i in valid_edges_set or i_inv in test_edges_set or i_inv in valid_edges_set:
            pass
        else:
            filt_ei.append(i)
    
    filt_ei_set = set(filt_ei)
    for i in train_edges:
        if i not in filt_ei_set:
            filt_ei.append(i)

    edge_index = torch.Tensor(filt_ei)
    graph_dict['edge_index'] = to_undirected(coalesce(edge_index.T.long()))
    graph_dict['split_edge']['train'] = torch.tensor(train_edges)
    graph_dict['split_edge']['valid'] = torch.tensor(valid_edges)
    graph_dict['split_edge']['test'] = torch.tensor(test_edges)

    print("Final Train/Valid/Test:", graph_dict['split_edge']['train'].shape, 
          graph_dict['split_edge']['valid'].shape, 
          graph_dict['split_edge']['test'].shape)

    dataset_path = f"dataset/{dataset}Dataset/" #Explicit PATH for easier checking
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    data = fill_data_feats(graph_dict)

    return data, graph_dict

def save_samples(samples, file_name):
    print("Saving samples...")
    with open(file_name, "wb") as f:
        np.save(f, samples)


def gen_edge_samps(data, graph_dict, tvt, args):
    # Simulates citations between nodes for edge/degree split functions
    #For negative sampling - k = 250, take sets for k/2 - (a, *) and (*, b). 
    # sample code then takes row from adjacency matrix and then sample directly from there. 
    k = args.num_samples // 2
    split_dict = {'edge': [], 'edge_neg': torch.tensor([[torch.tensor(0),torch.tensor(0)]])}
    pos_edge = []
    adj = data.adj_t
    tot_rand_edges = []

    for i in graph_dict['split_edge'][tvt]:
        if len(pos_edge) >= 3000000: break # Limit to 3 mil training samples
        if args.inverse and 'ddi' in args.data_name and len(pos_edge) >= 10000: break
        if tvt == 'train':
            assert len(data['adj_t'][i[0].item(), i[1].item()].storage.value()) == 1, f"Train Edge, {i} not found in A/edge_index"
        else:
            assert len(data['adj_t'][i[0].item(), i[1].item()].storage.value()) == 0, f"{ tvt.upper() } Edge, {i} found in A/edge_index"
        
        pos_edge.append(i)

    split_dict['edge'] = torch.stack(pos_edge)

    if tvt != 'train':
        print(f"{ tvt.upper()} Edge Split")
        #Limit number of positive samples to 100k
        if split_dict['edge'].size(0) > 100000:
            idx = torch.randperm(split_dict['edge'].size(0))[:100000]
            split_dict['edge'] = split_dict['edge'][idx]
        # Random Negative Edge Sampling Loop
        for edge in split_dict['edge']:
            source, target = edge[0].item(), edge[1].item()
           
            source_adj = adj[source].to_dense().squeeze()
            target_adj = adj[target].to_dense().squeeze()
            
            # Set Self-Loops
            source_adj[source], source_adj[target] = 1, 1
            target_adj[target], target_adj[source] = 1, 1
   
            source_adj = torch.nonzero(source_adj == 0).to_dense().squeeze()
            target_adj = torch.nonzero(target_adj == 0).to_dense().squeeze()
            # Currently passes elements instead of indices -- need indices for proper negative samples that have zero elements (i.e. an incorrect LP guess)  
            source_edges = np.concatenate((np.repeat(source, k).reshape(-1, 1), np.random.choice(source_adj, k).reshape(-1, 1)), axis=-1)
            target_edges = np.concatenate((np.random.choice(target_adj, k).reshape(-1, 1), np.repeat(target, k).reshape(-1, 1)), axis=-1)
            rand_sam_edges = np.concatenate((source_edges, target_edges))
            tot_rand_edges.append(rand_sam_edges)
        tot_rand_edges = np.stack(tot_rand_edges)
    else:
        tot_rand_edges = np.stack(np.zeros((1, 1)))
        print('Train Edge Split')

    split_dict['edge_neg'] = torch.tensor(tot_rand_edges)    

    return split_dict, tot_rand_edges

def build_ddi_feat(seed, data):
    torch.manual_seed(42) 
    ddi_feat = torch.nn.Embedding(data.num_nodes, 128)
    torch.nn.init.xavier_uniform_(ddi_feat.weight)
    ddi_feat.weight.requires_grad = False
    data.x = ddi_feat(torch.arange(data.num_nodes))
    print("ddi node features: ", data.x)
    init_seed(seed)

    return data

def build_split(dataset_name, graph_dict, args):
    data = torch.load(f"dataset/{dataset_name}Dataset/{dataset_name}Dataset.pt")
    train, _ = gen_edge_samps(data, graph_dict, 'train', args)
    valid, _ = gen_edge_samps(data, graph_dict, 'valid', args)
    test, _ = gen_edge_samps(data, graph_dict, 'test', args)

    split_edge = {'train': train, 'valid': valid, 'test': test}
    calc_all_heuristics(args, data, split_edge, dataset_name)

    return split_edge

def build_dataset(dataset_name, seed, args):
    dataset_name = dataset_name + f'_seed{seed}'
    init_seed(seed)
    print(f"Building {dataset_name}")
    data, graph_dict=build_synth(dataset_name, args=args)
    if 'ddi' in args.data_name.lower(): data = build_ddi_feat(seed, data)
    torch.save(data, f"dataset/{dataset_name}Dataset/{dataset_name}Dataset.pt")
    data_split=build_split(dataset_name, graph_dict, args)
    torch.save(data_split, f"dataset/{dataset_name}Dataset/{dataset_name}Dataset_split.pt")
    
def main():
    parser = argparse.ArgumentParser(description='Synthetic Graph Generation and Splitting for OOD')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--train_rat', type=str, default='0')
    parser.add_argument('--valid_rat', type=str, default='1')
    parser.add_argument('--test_rat', type=str, default='2')
    parser.add_argument('--split_type', type=str, default='cn')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=250)
    parser.add_argument("--metric", help="Either 'CN', 'SP', 'PA'", type=str, default="CN")
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--debug_heur', action='store_true', default=False)

    # For PPR
    parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=5e-5)
    parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)

    args = parser.parse_args()
    args.split_type = args.split_type.lower()
    if 'ogbl' in args.data_name:
        args.num_runs = 1
    if args.split_type == 'sp':
        args.train_rat = '0.0'
    print("current runs: ", str(args.num_runs))
    print("current num_samples", str(args.num_samples))

    valid_thres, test_thres = '', ''

    if '.' in args.train_rat and '.' in args.valid_rat and '.' in args.test_rat:
        assert args.split_type != 'cn', "Split Type must be SP if working with float ratios and not CN or PA"
        print(args.valid_rat.split('.'))
        print(args.test_rat.split('.'))
        valid_thres = '0' + args.valid_rat.split('.')[-1]
        test_thres = '0' + args.test_rat.split('.')[-1]
        train_thres = '0' + args.train_rat.split('.')[-1]
        args.train_rat = float(args.train_rat)
        args.valid_rat = float(args.valid_rat)
        args.test_rat = float(args.test_rat)
    elif '.' not in args.train_rat and '.' not in args.valid_rat and '.' not in args.test_rat:
        valid_thres = args.valid_rat
        test_thres = args.test_rat
        train_thres = args.train_rat
        args.train_rat = int(args.train_rat)
        args.valid_rat = int(args.valid_rat)
        args.test_rat = int(args.test_rat)
    else:
        raise Exception("Invalid Split values, SP = float, (CN, PA) = int")

    if args.inverse:
        dataset_name = f'{args.data_name}_{args.split_type.upper()}_{test_thres}_{valid_thres}_{train_thres}'
    else:
        dataset_name = f'{args.data_name}_{args.split_type.upper()}_{train_thres}_{valid_thres}_{test_thres}'

    for i in range(1, args.num_runs+1):
        build_dataset(dataset_name, i, args)

if __name__ == "__main__":
    main()
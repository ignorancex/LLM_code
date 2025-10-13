# pylint: skip-file
import os
import sys
import numpy as np
import networkx as nx
import random
import gc
import torch
from datetime import datetime
import torch.optim as optim
from itertools import chain
from extensions.synthetic_gen.data_creator import *



### Functions for Edge Feats 
def process_edge_feats(args, graphs):
    '''
    Generates edge features and indices needed for training.
    
    Arguments:
        - args (Namespace): command line arguments
        - graphs (List[nx.Graph]): Networkx graphs used for training
    
    Output:
        - list_edge_feats (List[Tensor] or None): 
             list of scaled edge feats for each graph (device: args.device).
        - list_num_edges (List[Int]): 
             list of number of edges in each graph
        - lv_lists (List[Int] or None): 
             base indexing lists for Fenwick Weight Tree (BiGG-E)
        - list_last_edges (List[Int] or None):
             list of last edge index in each row of the adjacency matrix (BiGG-E)
    '''
    list_num_edges = [len(g.edges()) for g in graphs]
    if not args.has_edge_feats:
        return None, list_num_edges, None, None
    
    list_edge_feats = [args.wt_scale * torch.from_numpy(get_edge_feats(g)).to(args.device) for g in graphs]    
    lv_lists, list_last_edges = None, None
            
    if args.sampling_method == "softplus": 
        list_edge_feats = [compute_softminus(x) for x in list_edge_feats]
    
    elif args.sampling_method == "lognormal":
        list_edge_feats = [torch.log(x) for x in list_edge_feats]
    
    if args.method == "BiGG-E":
        lv_lists = [get_single_lv_list(num_edges) for num_edges in list_num_edges]
        list_last_edges = [get_last_edge(g) for g in graphs]
    
    return list_edge_feats, list_num_edges, lv_lists, list_last_edges


def batch_edge_info(batch_indices, list_edge_feats=None, lv_lists=None, list_last_edges=None, list_num_edges=None):
    '''
    Provides indices for edge featutes and indices in the current batch.
    
    Arguments:
        - batch_indices (List[Int]): indices of graphs in current batch
        - list_edge_feats (List[Tensor] or None): list of edge feature tensors for all graphs
        - lv_lists (List[Int] or None): list of base indices for Fenwick Weight Tree (BiGG-E)
        - list_last_edges (List[np.darray] or None): list of arrays of the last edge index in each row of the adjacency matrix (BiGG-E)
        - list_num_edges (List[Int] or None): list of number of edges in each graph
    
    Output:
        - edge_info (Dict[str, Any]):  Dictionary with the following elements:
            - 'edge_feats' (Tensor or None): tensor of edge features in current batch
            - 'batch_num_edges' (List[Int] or None): number of edges for each graph in the batch 
            - 'first_edge' (List[Int] or None): offset index for first row edges 
            - 'cur_lv_lists' (List[Int] or None): Fenwick Weight Tree indices for the batch
            - 'batch_last_edges' (List[Int] or None): list of last edges in each row offset for the batch
    '''
    edge_info = {'edge_feats': None, 'batch_last_edges': None, 'cur_lv_lists': None, 'batch_num_edges': None, 'first_edge': None}
    
    if list_edge_feats is None:
        return edge_info
    
    edge_info['edge_feats'] = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
    
    if lv_lists is None:
        return edge_info
    
    edge_info['batch_num_edges'] = [list_num_edges[i] for i in batch_indices]
    first_edge = [0]
    for n in edge_info['batch_num_edges'][:-1]:
        first_edge.append(first_edge[-1] + n)
    
    edge_info['first_edge'] = first_edge
    edge_info['cur_lv_lists'] = [lv_lists[i] for i in batch_indices]
    
    batch_last_edges = [list_last_edges[i] for i in batch_indices]
    offset = 0
    for b, batch_last_edges_b in enumerate(batch_last_edges):
        mask_b = (batch_last_edges_b != -1)
        batch_last_edges_b_offset = np.zeros_like(batch_last_edges_b)
        batch_last_edges_b_offset[mask_b] = offset
        batch_last_edges[b] = batch_last_edges_b + batch_last_edges_b_offset
        offset += edge_info['batch_num_edges'][b]
    edge_info['batch_last_edges'] = np.concatenate(batch_last_edges, axis = 0)
    return edge_info


def compute_softminus(edge_feats, threshold = 20):
    '''
    Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
    reverts to linear function if w > 20.
    
    Arguments:
        - edge_feats [Tensor]: tensor of edge features
        - threshold (float): threshold value to revert to linear function
    
    Output:
        - Tensor: edge features with softminus function applied
    '''
    x_thresh = (edge_feats <= threshold).float()
    x_sm = torch.log(torch.special.expm1(edge_feats))
    x_sm = torch.mul(x_sm, x_thresh)
    x_sm = x_sm + torch.mul(edge_feats, 1 - x_thresh)
    return x_sm


def t(n1, n2):
    '''
    Returns the index of the (n1, n2)th entry of the adjacency matrix when
    traversing the lower half by rows
    
    Arguments:
        - n1 (int): First node in edge
        - n2 (int): Second node in edge
    
    Outputs:
        - t (int): index of the edge
    '''
    r = max(n1, n2)
    c = min(n1, n2)
    t = r * (r - 1) // 2 + c
    return t


def get_edge_feats(g):
    '''
    Provides list of edge features in the order of traversing the lower half
    of the adjacency matrix by rows
    
    Arguments:
        - g [Networkx]: inputted graph
    
    Outputs:
       - weights [np.array, float32]: edge features in given order
    
    '''
    edges = sorted(g.edges(data=True), key=lambda x: t(x[0], x[1]))
    weights = [x[2]['weight'] for x in edges]
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)
    

def get_last_edge(g):
    '''
    Provides the index of the last edge found in each row of the lower half 
    of the adjacency matrix of g.
    
    Arguments
        - g [Networkx]: inputted graph
    
    Outputs:
        - last_edges [list]: list of last edge indices for each row.
    
    '''
    last_edges = []
    idx = -1
    idx_count = -1
    for r in sorted(g.nodes()):
        neighbors = [n for n in list(g.neighbors(r)) if n < r]
        idx_count += len(neighbors)
        if len(neighbors) > 0:
            c = max(neighbors)
            idx = idx_count
            last_edges.append(idx)
        else:
            if r == 0:
                last_edges.append(-1)
            else:
                last_edges.append(last_edges[-1])
    
    last_edges = [-1] + last_edges[:-1]
    return np.array(last_edges)






### Functions for Weight Fenwick Tree
def get_list_edge(cur_nedge_list):
    cur_nedge_array = np.array(cur_nedge_list, dtype=np.int32)
    nedge2_array = cur_nedge_array - (cur_nedge_array % 2)

    # Calculate starting offset for each "block"
    offsets = np.cumsum(np.insert(cur_nedge_array[:-1], 0, 0))

    # Calculate ranges for each block
    edge_ranges = [np.arange(offset, offset + nedge2, dtype=np.int32)
                   for offset, nedge2 in zip(offsets, nedge2_array)]

    # Concatenate all ranges
    list_edge = np.concatenate(edge_ranges).tolist()
    return list_edge


def get_list_indices(nedge_list):
    '''Retrieves list of indices for states of batched graphs'''
    max_lv = int(np.log2(max(nedge_list)))
    list_indices = []
    list_edge = get_list_edge(nedge_list)
    cur_nedge_list = np.array(nedge_list, dtype=np.int32)
    empty = np.array([], dtype=np.int32)
    
    for lv in range(max_lv):
        left = np.array(list_edge[0::2], dtype=np.int32)
        right = np.array(list_edge[1::2], dtype=np.int32)
        
        left_indices = np.arange(len(left), dtype=np.int32)
        right_indices = np.arange(len(right), dtype=np.int32)
        
        # Append tuples for left and right
        list_indices.append([
            (empty, empty, left, left_indices, empty, empty),
            (empty, empty, right, right_indices, empty, empty)])
        
        cur_nedge_list = np.right_shift(cur_nedge_list, 1)  # Efficient halving
        list_edge = get_list_edge(cur_nedge_list.tolist())
    
    return list_indices


def lv_offset(num_edges, max_lv):
    offset_list = []
    while num_edges >= 1:
        offset_list.append(num_edges)
        num_edges >>= 1
    offset_list = np.pad(offset_list, (0, max_lv - len(offset_list)), 'constant', constant_values=0)
    num_entries = np.sum(offset_list)
    return offset_list, num_entries

def batch_lv_list(k, list_offset):
    lv_list = []
    bin_len = len(bin(k)) - 2
    offset_batch_cumsum = np.cumsum(list_offset, axis=0)

    for i in range(bin_len):
        if (k >> i) & 1:
            offset_tot = np.sum(list_offset[:, :i]) if i > 0 else 0
            val = k // (1 << i) + offset_tot - 1
            
            if i < list_offset.shape[1]:
                offset_batch = np.zeros(list_offset.shape[0], dtype=np.int32)
                offset_batch[1:] = np.cumsum(list_offset[:-1, i])
                offset_batch = offset_batch[list_offset[:, 0] >= k]
                val = val + offset_batch

            lv_list.append(val)

    # Stack all collected values
    lv_list = np.stack(lv_list, axis=1)
    return lv_list


def get_batch_lv_list_fast(list_num_edges): 
    max_lv = int(np.max(np.log2(list_num_edges)) + 1)
    
    # Precompute list_offset for each graph
    list_offset = np.array([lv_offset(num_edges, max_lv)[0] for num_edges in list_num_edges])

    max_edge = np.max(list_num_edges)
    batch_size = len(list_num_edges)

    # Initialize output as a list of lists
    out = [[] for _ in range(batch_size)]
    num_edges_array = np.array(list_num_edges, dtype=np.int32)

    for k in range(1, max_edge + 1):
        cur = (k <= num_edges_array)
        cur_lvs = batch_lv_list(k, list_offset)
        
        i = 0
        for batch, active in enumerate(cur):
            if active:
                out[batch].append(cur_lvs[i].tolist())
                i += 1

    return out


def get_batch_lv_lists(list_num_edges, lv_list):
    max_lv = int(np.max(np.log2(list_num_edges)) + 1)
    
    # Compute the per-graph level offsets
    list_offset = np.array([lv_offset(num_edges, max_lv)[0] for num_edges in list_num_edges])

    # Compute the cumulative offsets matrix for each graph and level
    mat = list_offset.T.flatten()[:-1]
    mat = np.insert(mat, 0, 0)
    list_offset_cumsum = np.cumsum(mat).reshape(list_offset.T.shape).T

    max_edge = np.max(list_num_edges)
    batch_size = len(list_num_edges)
    
    # Precompute binary representations and indices of '1' bits
    max_n = list(range(1, max_edge + 1))
    binary_digits = [bin(x)[2:] for x in max_n]
    max_indices_rtl = [[i for i, bit in enumerate(b[::-1]) if bit == '1'] for b in binary_digits]
    
    # Adjust each graph's lv_list using the cumulative offsets
    out = []
    for lv_idx, (lv, num_edges) in enumerate(zip(lv_list, list_num_edges)):
        indices_rtl = max_indices_rtl[:num_edges]
        offset_batch_cumsum = list_offset_cumsum[lv_idx, :]

        # Offset each entry of the lv_list by the correct batch offset
        lv_offset_new = [(x + offset_batch_cumsum[y]).tolist() for x, y in zip(lv, indices_rtl)]
        out.append(lv_offset_new)
    return out


def get_single_lv_list(num_edges):
    max_lv = int(np.log2(num_edges)) + 1
    offset_list, _ = lv_offset(num_edges, max_lv)

    out = []
    for k in range(1, num_edges + 1):
        lv_list = []
        bin_len = len(bin(k)) - 2
        for i in range(bin_len):
            if (k >> i) & 1:
                val = int(k // (1 << i) - 1)
                lv_list.append(val)
        out.append(lv_list)
    return out

def prepare_batch(batch_lv_in):
    batch_size = len(batch_lv_in)
    list_num_edges = [len(lv_in) for lv_in in batch_lv_in]
    tot_num_edges = np.sum(list_num_edges)
    
    # Flatten once
    flat_lv_in = list(chain.from_iterable(batch_lv_in))
    list_lvs = [len(l) for lv_in in batch_lv_in for l in lv_in]
    max_len = max(list_lvs)

    all_ids = []
    init_select = [x[0] for lv_in in batch_lv_in for x in lv_in]
    last_tos = [j for j, l in enumerate(list_lvs) if l == max_len]

    lens = np.array([len(l) for l in flat_lv_in])

    lv = 1
    while True:
        done_from = np.where(lens == 1)[0].tolist()
        done_to = [j for j, l in enumerate(list_lvs) if l == lv]

        proceed_indices = np.where(lens > 1)[0]
        proceed_from = proceed_indices.tolist()
        proceed_input = [flat_lv_in[i][1] for i in proceed_indices]

        all_ids.append((done_from, done_to, proceed_from, proceed_input))

        # Remove the first element for proceed
        flat_lv_in = [flat_lv_in[i][1:] for i in proceed_indices]
        lens = lens[lens > 1] - 1

        lv += 1
        if np.max(lens, initial=0) <= 1:
            break

    return init_select, all_ids, last_tos


### GCN Batching
def GCNN_batch_train_graphs(train_graphs, batch_indices, args):
    batch_g = nx.Graph()
    feat_idx = torch.Tensor().to(args.device)
    batch_weight_idx = []
    edge_list = []
    offset = 0
    for idx in batch_indices:
        g = train_graphs[idx]
        n = len(g)
        feat_idx = torch.cat([feat_idx, torch.arange(n).to(args.device)])
        for e1, e2, w in g.edges(data=True):
            batch_weight_idx.append((int(e1), int(e2), w['weight'] * args.wt_scale))
            edge_list.append((int(e1) + offset, int(e2) + offset, idx))
        offset += n
    edge_idx = torch.Tensor(edge_list).to(args.device).t()
    batch_weight_idx = torch.Tensor(batch_weight_idx).to(args.device)
    return feat_idx, edge_idx, batch_weight_idx


def reset_all_seeds(seed=42):
    '''
    Resets seeds for reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
## Debug Functions
def debug_model(model, train_graphs, list_edge_feats=None, list_last_edges=None, batch_indices = [0, 1], lv_lists = None, args=None):
    if args.has_edge_feats and args.method == "BiGG-E":    
        edge_feats = [list_edge_feats[i] for i in batch_indices]
        unbatched_last_edges =  [list_last_edges[i] for i in batch_indices]
        batch_last_edges = [list_last_edges[i] for i in batch_indices]
        graphs = [train_graphs[i] for i in batch_indices]
        cur_lv_lists = [lv_lists[i] for i in batch_indices]
        offset = 0
        
        for b in range(len(batch_last_edges)):
            batch_last_edges[b] = np.array([x + offset if x != -1 else x for x in batch_last_edges[b]])
            offset += len(train_graphs[batch_indices[b]].edges())
        batch_last_edges = np.concatenate(batch_last_edges)
                
        debug_model_2(model, graphs, edge_feats, batch_last_edges, batch_indices, unbatched_last_edges = unbatched_last_edges, lv_lists = cur_lv_lists, args=args)
    
    elif args.has_edge_feats and args.method != "BiGG-GCN":
        graphs = [train_graphs[i] for i in batch_indices]
        edge_feats = [list_edge_feats[i] for i in batch_indices]
        debug_model_2(model, graphs, edge_feats, batch_indices = batch_indices, args=args)
    
    else:
        graphs = [train_graphs[i] for i in batch_indices]
        debug_model_2(model, graphs, batch_indices = batch_indices, args=args)


def debug_model_2(model, graph, edge_feats=None, batch_last_edges=None, batch_indices = [0, 1], unbatched_last_edges = None, lv_lists = None, args=None):
    ll_t1 = 0
    ll_w1 = 0
    ll_t2 = 0
    ll_w2 = 0
    ll_t3 = 0
    ll_w3 = 0
    threshold = 10.0  # Change this as needed
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            max_abs = data.abs().max().item()
            if max_abs > threshold:
                mean = data.mean().item()
                std = data.std().item()
                print(f"{name}: max |param| = {max_abs:.4f}, mean = {mean:.4f}, std = {std:.4f}")
    
    if edge_feats is not None and args.method == "BiGG-E":
        list_num_edges = [len(x) for x in edge_feats]
    
    else:
        list_num_edges = None
    
    h_fast = []
    c_fast = []
    h_slow = []
    c_slow = []
    model.eval()
    torch.set_printoptions(threshold=float('inf'))
    for idx,i in enumerate(batch_indices):
        g = graph[idx]
        edge_feats_i = (edge_feats[idx] if edge_feats is not None else None)
        batch_last_edges_i = (np.array(unbatched_last_edges[idx]) if unbatched_last_edges is not None else None)
        list_num_edges_i = ([list_num_edges[idx]] if list_num_edges is not None else None)
        lv_lists_i = ([lv_lists[idx]] if lv_lists is not None else None)
        first_edge = ([0] if list_num_edges_i is not None else None)        
        edges = []
        h_diffs = 0.0
        c_diffs = 0.0
        for e in g.edges():
            if e[1] > e[0]:
                e = (e[1], e[0])
            edges.append(e)
        edges = sorted(edges)
        
        if edge_feats_i is not None and not torch.is_tensor(edge_feats_i):
            edge_feats_i = edge_feats_i[0]
        
        with torch.no_grad():
            if args.method == "BiGG-GCN":
                reset_all_seeds(args.seed)
                ll, _, _, row_states_slow, _ = model(len(g), edges, edge_feats=edge_feats_i)
                row_states_slow = (row_states_slow[0].clone().detach().cpu(), row_states_slow[1].clone().detach().cpu())
                reset_all_seeds(args.seed)
                _, _, _, row_states_slow_2, _ = model(len(g), edges, edge_feats=edge_feats_i)
                row_states_slow_2 = (row_states_slow_2[0].clone().detach().cpu(), row_states_slow_2[1].clone().detach().cpu())
                ll_wt = 0.0
                reset_all_seeds(args.seed)
                ll2, row_states_fast = model.forward_train([i])
                row_states_fast= (row_states_fast[0].clone().detach().cpu(), row_states_fast[1].clone().detach().cpu())
                ll_wt2 = 0.0
                reset_all_seeds(args.seed)
                _, row_states_fast_2 = model.forward_train([i])
                row_states_fast_2 =  (row_states_fast_2[0].clone().detach().cpu(), row_states_fast_2[1].clone().detach().cpu())
                
                print("Checking norms on identical FAST passes")
                print(torch.norm(row_states_fast[0] - row_states_fast_2[0]))
                print(torch.norm(row_states_fast[1] - row_states_fast_2[1]))
                print("Checking norms on identical SLOW passes")
                print(torch.norm(row_states_slow[0] - row_states_slow_2[0]))
                print(torch.norm(row_states_slow[1] - row_states_slow_2[1]))
                print("++++++++++++++++++++++++++++++++++++++++")
            
            else:
                reset_all_seeds(args.seed)
                ll, ll_wt, _, _, row_states_slow, _ = model(len(g), edges, edge_feats=edge_feats_i)
                edge_info = {'edge_feats': edge_feats_i, 'batch_last_edges': batch_last_edges_i, 'cur_lv_lists': lv_lists_i, 'batch_num_edges': list_num_edges_i, 'first_edge': first_edge}
                reset_all_seeds(args.seed)
                ll2, ll_wt2, row_states_fast = model.forward_train([i], edge_feat_info = edge_info)
                
                reset_all_seeds(args.seed)
                _, _, row_states_fast_2 = model.forward_train([i], edge_feat_info = edge_info)
                
                print("Checking norms on identical passes")
                print(torch.norm(row_states_fast[0] - row_states_fast_2[0]))
                print(torch.norm(row_states_fast[1] - row_states_fast_2[1]))
                print("++++++++++++++++++++++++++++++++++++++++")
        
        if True:
            h_slow.append(row_states_slow[0])
            h_fast.append(row_states_fast[0])
            c_slow.append(row_states_slow[1])
            c_fast.append(row_states_fast[1])
        
        ll_t2 = ll + ll_t2
        ll_w2 = ll_wt + ll_w2
        
        ll_t3 = ll2 + ll_t3
        ll_w3 = ll_wt2 + ll_w3
    
    if isinstance(edge_feats, list):
        edge_feats = torch.cat(edge_feats, dim = 0)
    
    first_edge = None
    if list_num_edges is not None:
        first_edge = [0]
        for n in list_num_edges[:-1]:
            first_edge.append(first_edge[-1] + n)
    
    with torch.no_grad():
        if args.method == "BiGG-GCN":
            reset_all_seeds(args.seed)
            ll_t1, row_states_batch = model.forward_train(batch_indices)
            ll_w1 = 0.0
            row_states_batch = (row_states_batch[0].clone().detach().cpu(), row_states_batch[1].clone().detach().cpu())
            reset_all_seeds(args.seed)
            _, row_states_batch_2 = model.forward_train(batch_indices)
            row_states_batch_2 = (row_states_batch_2[0].clone().detach().cpu(), row_states_batch_2[1].clone().detach().cpu())
        
        else:
            edge_info = {'edge_feats': edge_feats, 'batch_last_edges': batch_last_edges, 'cur_lv_lists': lv_lists, 'batch_num_edges': list_num_edges, 'first_edge': first_edge}
            reset_all_seeds(args.seed)
            ll_t1, ll_w1, row_states_batch = model.forward_train(batch_indices, edge_feat_info=edge_info)
            
            reset_all_seeds(args.seed)
            _, _, row_states_batch_2 = model.forward_train(batch_indices, edge_feat_info=edge_info)
    
    if True:
        print("==============================")
        print("Checking Hidden and Cell Row State Differences")
        h_fast_first = h_fast[0]
        c_fast_first = c_fast[0]
        h_fast_second = h_fast[1]
        c_fast_second = c_fast[1]
        num = h_fast_first.shape[0]
        h_fast = torch.cat(h_fast, dim = 0)
        h_slow = torch.cat(h_slow, dim = 0)
        c_fast = torch.cat(c_fast, dim = 0)
        c_slow = torch.cat(c_slow, dim = 0)
        
        print(h_fast.shape)
        print("Normed differences **summed across all rows** (entire graph state):")
        x = torch.norm(h_slow - row_states_batch[0], dim=1)
        print("Norm BATCH VS SLOW H" , "with mean", x.mean(), "max abs", x.abs().max())
        x = torch.norm(c_slow - row_states_batch[1], dim=1)
        print("Norm BATCH VS SLOW C",  "with mean", x.mean(), "max abs", x.abs().max())
    
        x = torch.norm(h_fast - row_states_batch[0], dim=1)
        print("Norm BATCH VS UNBATCH H",  "with mean", x.mean(), "max abs", x.abs().max())
        x = torch.norm(c_fast - row_states_batch[1], dim=1)
        print("Norm BATCH VS UNBATCH C",  "with mean", x.mean(), "max abs", x.abs().max())
        
        x = torch.norm(h_slow - h_fast, dim=1)
        print("Norm UNBATCH VS SLOW H", "with mean", x.mean(), "max abs", x.abs().max())
        x = torch.norm(c_slow - c_fast, dim=1)
        print("Norm UNBATCH VS SLOW C", "with mean", x.mean(), "max abs", x.abs().max())
        
        c_slow_last = c_slow[-1, :]
        c_fast_last = c_fast[-1, :]
        c_batch_last = row_states_batch[1][-1, :]
        print("Now, we are ONLY looking at the **final cell state** (last node) of a **single graph**")
        print("C slow vs fast: ", torch.norm(c_slow_last - c_fast_last))
        print("C slow vs batch: ", torch.norm(c_slow_last - c_batch_last))
        print("C batch vs fast: ", torch.norm(c_batch_last - c_fast_last))
        
        print("Finally, comparing **row-wise norm difference for graph batch [0]** versus the **first graph in batch [0, 1]**")
        print("First Graph")
        print("Hidden Norm: ", torch.norm(h_fast_first - row_states_batch[0][0:num, :]))
        print("Cell Norm: ", torch.norm(c_fast_first - row_states_batch[1][0:num, :]))
        print("Second Graph")
        print("Hidden Norm: ", torch.norm(h_fast_second - row_states_batch[0][num:, :]))
        print("Cell Norm: ", torch.norm(c_fast_second - row_states_batch[1][num:, :]))
        
        if batch_indices[0] == batch_indices[1]:
            print("Compare batched case with identical graphs")
            print("Hidden norm: ", torch.norm(row_states_batch[0][0:num, :] - row_states_batch[0][num:, :]))
            print("Cell norm: ", torch.norm(row_states_batch[1][0:num, :] - row_states_batch[1][num:, :]))
        
        print("Testing consistency in forward passes")
        print("Hidden norm: ", torch.norm(row_states_batch[0] - row_states_batch_2[0]))
        print("Cell norm: ", torch.norm(row_states_batch[1] - row_states_batch_2[1]))
        
    print("=============================")
    print("=============================")
    print("Fast Code Top+Wt Likelihoods: ")
    print(ll_t1)
    print(ll_w1)
    print("=============================")
    print("Slow Code Top+Wt Likelihoods: ")
    print(ll_t2)
    print(ll_w2)
    print("=============================")
    print("Unbatched Fast Code Top+Wt Likelihoods: ")
    print(ll_t3)
    print(ll_w3)
    print("=============================")
    
    diff1 = abs(ll_t1 - ll_t2)
    diff2 = abs(ll_w1 - ll_w2)
    diff3 = abs(ll_t1 - ll_t3)
    diff4 = abs(ll_w1 - ll_w3)
    diff5 = abs(ll_t2 - ll_t3)
    diff6 =abs(ll_w2 - ll_w3)

    print("Absolute Differences Between Fast and Slow Code: ")
    print("diff top: ", diff1)
    print("diff weight: ", diff2)
    print("=============================")
    
    rel1 = (ll_t1 - ll_t2) / (ll_t1 + 1e-15)
    rel2 = (ll_w1 - ll_w2) / (ll_w1 + 1e-15)
    
    print("Relative Differences (%): ")
    print("rel diff top: ", rel1 * 100)
    print("rel diff weight: ", rel2 * 100)
    print("=============================")
    
    print("Absolute Differences Between Batched and Unbatched Code: ")
    print("diff top: ", diff3)
    print("diff weight: ", diff4)
    print("=============================")
    
    rel3 = (ll_t1 - ll_t3) / (ll_t1 + 1e-15)
    rel4 = (ll_w1 - ll_w3) / (ll_w1 + 1e-15)
    
    print("Relative Differences (%): ")
    print("rel diff top: ", rel3 * 100)
    print("rel diff weight: ", rel4 * 100)
    print("=============================")
    
    print("Absolute Differences Between Slow and Unbatched Code: ")
    print("diff top: ", diff5)
    print("diff weight: ", diff6)
    print("=============================")
    
    rel5 = (ll_t2 - ll_t3) / (ll_t2 + 1e-15)
    rel6 = (ll_w2 - ll_w3) / (ll_w2 + 1e-15)
    
    print("Relative Differences (%): ")
    print("rel diff top: ", rel5 * 100)
    print("rel diff weight: ", rel6 * 100)
    print("=============================")
    
    import sys
    sys.exit()

## Training Helper Functions

def log_named_grad_norms(model, norm_type=2, threshold=10.0):
    print("== Gradient Norms Per Module ==")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(norm_type).item()
            flag = "ALERT" if grad_norm > threshold else ""
            print(f"{name:<50} | grad norm: {grad_norm:.4f} {flag}")


def get_total_grad_norm(params, norm_type=2):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
    
    
def process_graphs(graphs, node_order):
    from extensions.synthetic_gen.data_util import get_graph_data
    assert len(graphs) == 100
    ordered_graphs = []
    for g in graphs:
        cano_g = get_graph_data(g, node_order, global_source=0)
        ordered_graphs += cano_g
    
    num_train = 80
    num_test_gt = 20
    
    train_graphs = ordered_graphs[:num_train]
    test_graphs = ordered_graphs[num_train:]
    del graphs
    del ordered_graphs
    return train_graphs, test_graphs


def build_model_and_optimizers(args):
    from extensions.BiGG_E.model_extensions.customized_models import BiGGExtension, BiGGWithGCN
    if args.method == 'BiGG-GCN':
        args.has_edge_feats = False
        model = BiGGWithGCN(args).to(args.device)
        args.has_edge_feats = True
        
        topo_params = [p for n, p in model.named_parameters() if not n.startswith('gcn_mod.')]
        wt_params = [p for p in model.gcn_mod.parameters()]
    
    else:
        model = BiGGExtension(args).to(args.device)
        topo_params, wt_params = [], []
    
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue 
            if ("wt" in name and "proj_wt_c" not in name) or "proj_top_c" in name or "weight_tree" in name:
                wt_params.append(param)
            else:
                topo_params.append(param)
    
    params_list = [topo_params, wt_params]
    optimizer_topo = optim.AdamW(topo_params, lr=args.learning_rate_top, weight_decay=1e-4)
    decay_wt = (1e-4 if args.method == "BiGG-GCN" else 1e-2)
    optimizer_wt  = optim.AdamW(wt_params, lr=args.learning_rate_wt, weight_decay=decay_wt)
    
    if args.epoch_load != 0:
        model, optimizer_topo, optimizer_wt, params_list = load_model(args, model, optimizer_topo, optimizer_wt)
    
    return model, optimizer_topo, optimizer_wt, params_list
    

def load_model(args, model, optimizer_topo, optimizer_wt):
    # Load Model if Applicable
    base_path = os.path.join(args.base_path, args.method, args.g_type)
    
    if args.epoch_load == -1:
        print("Loading Most Recent Model Save")
        path = os.path.join(base_path, f'temp-{2 * args.num_leaves}-leaves.ckpt')
    
    else:
        print("Loading Epoch #: ", args.epoch_load)
        path = os.path.join(base_path, f'temp-{2 * args.num_leaves}-leaves-{args.epoch_load}.ckpt')
    
    if os.path.isfile(path):
        print('Loading Model')
        checkpoint = torch.load(path)
        
        ## Baseline Info
        model.load_state_dict(checkpoint['model'])
        optimizer_topo.load_state_dict(checkpoint['optimizer_topo'])
        optimizer_wt.load_state_dict(checkpoint['optimizer_wt'])
        args.learning_rate_top = checkpoint['learning_rate_top']
        args.learning_rate_wt = checkpoint['learning_rate_wt']
        if args.epoch_load == -1:
            args.epoch_load = checkpoint['epoch'] + 1
        
        ## If manually updating learning rates
        if args.learning_rate_top_update == 1e-3:
            for param_group in optimizer_topo.param_groups:
                param_group['lr'] = args.learning_rate_top
        
        else:
            args.learning_rate_top = args.learning_rate_top_update
            for param_group in optimizer_topo.param_groups:
                param_group['lr'] = args.learning_rate_top_update
        
        if args.learning_rate_wt_update == 1e-3:
            for param_group in optimizer_wt.param_groups:
                param_group['lr'] = args.learning_rate_wt
        
        else:
            args.learning_rate_wt = args.learning_rate_wt_update
            for param_group in optimizer_wt.param_groups:
                param_group['lr'] = args.learning_rate_wt_update
        
        wt_params, topo_params = [], []
        if args.method in ["BiGG-E", "BiGG-MLP"]:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue 
                if ("wt" in name and "proj_wt_c" not in name) or "proj_top_c" in name or "weight_tree" in name:
                    wt_params.append(param)
                else:   
                    topo_params.append(param)
        
        else:
            topo_params = [p for n, p in model.named_parameters() if not n.startswith('gcn_mod.')]
            wt_params = [p for p in model.gcn_mod.parameters()]
        
        params_list = [topo_params, wt_params]
    
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    return model, optimizer_topo, optimizer_wt, params_list


def build_model_and_optimizers_main(args):
    from extensions.BiGG_E.model_extensions.customized_models import BiGGExtension, BiGGWithGCN
    if args.method == 'BiGG-GCN':
        args.has_edge_feats = False
        model = BiGGWithGCN(args).to(args.device)
        args.has_edge_feats = True
        
        topo_params = [p for n, p in model.named_parameters() if not n.startswith('gcn_mod.')]
        wt_params = [p for p in model.gcn_mod.parameters()]
    
    else:
        model = BiGGExtension(args).to(args.device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        topo_params, wt_params = [], []
        topo_names, wt_names = [], []
    
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue 
            if ("wt" in name and "proj_wt_c" not in name) or "proj_top_c" in name or "weight_tree" in name:
                wt_params.append(param)
            else:
                topo_params.append(param)
    
    params_list = [topo_params, wt_params]
    optimizer_topo = optim.AdamW(topo_params, lr=args.learning_rate_top, weight_decay=1e-4)
    decay_wt = (1e-4 if args.method == "BiGG-GCN" else 1e-3)
    optimizer_wt  = (optim.AdamW(wt_params, lr=args.learning_rate_wt, weight_decay=decay_wt) if args.has_edge_feats else None)
    
    if args.model_dump is not None and os.path.isfile(args.model_dump):
        model, optimizer_topo, optimizer_wt, params_list = load_model_main(args, model, optimizer_topo, optimizer_wt)
    
    elif args.model_dump is not None:
        raise FileNotFoundError(f"Checkpoint not found: {args.model_dump}")
    
    return model, optimizer_topo, optimizer_wt, params_list
 

def load_model_main(args, model, optimizer_topo, optimizer_wt):
    print('loading from', args.model_dump)
    checkpoint = torch.load(args.model_dump)
    model.load_state_dict(checkpoint['model'])
    optimizer_topo.load_state_dict(checkpoint['optimizer_topo'])
    if optimizer_wt is not None:
        optimizer_wt.load_state_dict(checkpoint['optimizer_wt'])
    
    if args.learning_rate_top == 1e-3:
            args.learning_rate_top = checkpoint['learning_rate_top']
    
    if args.learning_rate_wt == 1e-3:
        args.learning_rate_wt = checkpoint['learning_rate_wt']
    if checkpoint['learning_rate_wt'] == 1e-5:
        args.scale_loss = 100
    
    for param_group in optimizer_topo.param_groups:
        param_group['lr'] = args.learning_rate_top
    if args.has_edge_feats:
        for param_group in optimizer_wt.param_groups:
            param_group['lr'] = args.learning_rate_wt
    
    wt_params, topo_params = [], []
    if args.method in ["BiGG-E", "BiGG-MLP"]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue 
            if ("wt" in name and "proj_wt_c" not in name) or "proj_top_c" in name or "weight_tree" in name:
                wt_params.append(param)
            else:   
                topo_params.append(param)
    
    elif args.method == "BiGG-GCN":
        topo_params = [p for n, p in model.named_parameters() if not n.startswith('gcn_mod.')]
        wt_params = [p for p in model.gcn_mod.parameters()]
    
    params_list = [topo_params, wt_params]
    return model, optimizer_topo, optimizer_wt, params_list


def param_step(model, params_list, args, optimizer_topo, optimizer_wt):
    topo_norm = get_total_grad_norm(params_list[0])
    wt_norm = get_total_grad_norm(params_list[1])
    tot_norm = get_total_grad_norm(params_list[0] + params_list[1])
    if topo_norm > 20 or wt_norm > 20:
        print(f"Topo Grad Norm: {topo_norm:.4f} | Weight Grad Norm: {wt_norm:.4f} | Total Grad Norm: {tot_norm:.4f}")
        log_named_grad_norms(model)
    if args.grad_clip > 0:
        if args.method == "BiGG-GCN":
            for params in params_list:
                torch.nn.utils.clip_grad_norm_(params, max_norm= args.grad_clip)
        
        else:
            torch.nn.utils.clip_grad_norm_(params_list[0] + params_list[1], max_norm= args.grad_clip)
             
    optimizer_topo.step()
    optimizer_topo.zero_grad()
    if optimizer_wt is not None:
        optimizer_wt.step()
        optimizer_wt.zero_grad()


def save_model(epoch, model, optimizer_topo, optimizer_wt, args, epoch_loss_top):
    print('Saving Model')
    checkpoint = {'epoch': epoch, 
                  'model': model.state_dict(), 
                  'optimizer_topo': optimizer_topo.state_dict(), 
                  'optimizer_wt': optimizer_wt.state_dict(), 
                  'learning_rate_top': args.learning_rate_top, 
                  'learning_rate_wt': args.learning_rate_wt}
    base_path = os.path.join(args.base_path, args.method, args.g_type)
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, f'temp-{2 * args.num_leaves}-leaves.ckpt')
    torch.save(checkpoint, path)
    
    cur = epoch+1
    if cur % 100 == 0 and epoch_loss_top < 1.0 :
        print('Saving Model for Epoch:', cur)
        path = os.path.join(base_path, f'temp-{2 * args.num_leaves}-leaves-{cur}.ckpt')
        checkpoint = {'epoch': epoch, 
                      'model': model.state_dict(), 
                      'optimizer_topo': optimizer_topo.state_dict(), 
                      'optimizer_wt': optimizer_wt.state_dict(), 
                      'learning_rate_top': args.learning_rate_top, 
                      'learning_rate_wt': args.learning_rate_wt}
        torch.save(checkpoint, path)


def precompute_batch_indices(indices, num_nodes_list, num_edges_list, args):
    grad_accum_counter = 0
    batch_dict = {}
    accum_grad = args.accum_grad
    B = args.batch_size
    num_iter = len(indices) // B
    
    for idx in range(num_iter):
        if grad_accum_counter == 0:
            start = accum_grad * B * idx // accum_grad
            stop = accum_grad * B * (idx // accum_grad + 1)
            tot_batch_indices = indices[start:stop]
            num_nodes = sum([num_nodes_list[i] for i in tot_batch_indices])
            num_edges = sum([num_edges_list[i] for i in tot_batch_indices]) 

        start = grad_accum_counter * B 
        stop = (grad_accum_counter + 1) * B
        batch_indices = tot_batch_indices[start:stop]
        cur_num_nodes = sum([num_nodes_list[i] for i in batch_indices])
        cur_num_edges = sum([num_edges_list[i] for i in batch_indices])   
        
        cur_idx_dict = {'start': start,
                        'stop': stop, 
                        'batch_indices': batch_indices,
                        'cur_num_nodes': cur_num_nodes,
                        'cur_num_edges': cur_num_edges,
                        'num_nodes': num_nodes,
                        'num_edges': num_edges}
        
        batch_dict[idx] = cur_idx_dict
        
        grad_accum_counter += 1
        if grad_accum_counter == accum_grad:
            grad_accum_counter = 0
        
    return batch_dict


def time_model_forward(method, args, num_nodes):
    args.max_num_nodes = num_nodes
    model, _, _, _ = build_model_and_optimizers(args)
    model.eval()
    import time
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        _, pred_edges, _, _, pred_edge_feats = model(node_end = num_nodes, display=args.display)
        if args.method == 'BiGG-GCN':
            fix_edges = [(min(e1, e2), max(e1, e2)) for e1, e2 in pred_edges]
            pred_edge_tensor = torch.tensor(fix_edges).to(args.device)
            pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.time() - start
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


def graph_in_dataset(generated_graph, training_graphs):
    """
    Checks if `generated_graph` is structurally isomorphic to any graph in `training_graphs`.
    Ignores weights or other attributes.
    """
    for i, train_graph in enumerate(training_graphs):
        if nx.is_isomorphic(generated_graph, train_graph):
            return True, i
    return False, -1
    

def extract_weights_with_stats(graphs):
    weight_lists = []
    mins, q1s, medians, q3s, maxs = [], [], [], [], []
    means, stds = [], []
    
    for G in graphs:
        weights = [data['weight'] for _, _, data in G.edges(data=True) if 'weight' in data]
        if not weights:
            # Handle graphs with no weighted edges
            weights = [np.nan]

        weights_np = np.array(weights)
        weight_lists.append(weights)

        mins.append(np.min(weights_np))
        q1s.append(np.percentile(weights_np, 25))
        medians.append(np.median(weights_np))
        q3s.append(np.percentile(weights_np, 75))
        maxs.append(np.max(weights_np))
        means.append(np.mean(weights_np))
        stds.append(np.std(weights_np, ddof=1))

    return {
        "weights": weight_lists,
        "min": mins,
        "q1": q1s,
        "median": medians,
        "q3": q3s,
        "max": maxs,
        "mean": means,
        "std": stds
    }


def set_debug_args(args):
    batch_indices = [0, 1]
    args.batch_size = len(batch_indices)
    args.sigma = 0.0
    args.wt_drop = -1
    args.phase = "train"
    if args.epoch_load is None:
        args.mu_0 = False
        args.dynam_score = False
    if args.method == "BiGG-GCN":
        args.has_edge_feats = False
    return batch_indices


def update_weight_stats(model, list_edge_feats, method):
    for i, edge_feats in enumerate(list_edge_feats):
        initialize=(i+1 == len(list_edge_feats))
        if method == "BiGG-GCN":
            model.gcn_mod.update_weight_stats(edge_feats)
        else:
            model.update_weight_stats(edge_feats, initialize)


def get_num_nodes(train_graphs, gt_graphs, num_node_dist, args):
    if False and args.g_type == "db":
        train_node_counts = sorted(set(len(g.nodes) for g in train_graphs))
        
        # Match each GT graph to the closest training graph node count
        num_nodes_list = []
        for g in gt_graphs:
            num_nodes_gt = len(g.nodes)
            closest = min(train_node_counts, key=lambda x: abs(x - num_nodes_gt))
            num_nodes_list.append(closest)
    
    else:
        np.random.seed(args.seed)  # Set global NumPy seed
        num_nodes_list = [np.random.choice(len(num_node_dist), p=num_node_dist) for _ in range(args.num_test_gen)]
    
    return num_nodes_list
























import networkx as nx
import numpy as np
import pickle as cp
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
import os
import sys
import argparse
from tqdm import tqdm
from extensions.synthetic_gen.data_creator import *
from extensions.synthetic_gen.data_util import *
from extensions.common.configs import cmd_args

cmd_opt = argparse.ArgumentParser(description='Argparser for syn_gen')
local_args, _ = cmd_opt.parse_known_args()
cmd_args.__dict__.update(local_args.__dict__)

### Processing Weighted Graphs
if cmd_args.g_type == 'db':
    if cmd_args.load_db:
        path = os.path.join(cmd_args.save_dir, 'train-graphs.pkl')
        with open(path, 'rb') as f:
            train_graphs = cp.load(f)

        path = os.path.join(cmd_args.save_dir, 'test-graphs.pkl')
        with open(path, 'rb') as f:
            test_graphs = cp.load(f)

        graphs = train_graphs + test_graphs
    else:
        data_dir = '../db'
        graphs = graph_load_batch(
            data_dir,
            min_num_nodes=0,
            max_num_nodes=10000,
            name='FIRSTMM_DB',
            node_attributes=True,
            graph_labels=True)
        new_graphs = []
        max_edge_feats = []
        list_edge_feats = []
        for g in graphs:
            g = nx.Graph(g)
            edge_feats = []
            for (e1, e2, w) in g.edges(data=True):
                c1 = np.array(g.nodes[e1]['feature'])
                c2 = np.array(g.nodes[e2]['feature'])
                d = np.sum((c1 - c2)**2)
                d = d**0.5
                g[e1][e2]['weight'] = d
                edge_feats.append(d)
            new_graphs.append(g)
            edge_feats = np.array(edge_feats)

        graphs = new_graphs
        #fixed_train_graphs = []
        for g in graphs:
            dupes = []
            for (e1, e2, w) in g.edges(data=True):
                weight = w['weight']
                if weight == 0:
                    ## Duplicate nodes
                    if len(list(g.neighbors(e1))) >= len(list(g.neighbors(e2))):
                        dupes.append(e2)
                    else:
                        dupes.append(e1)
            print("Num dupes: ", len(dupes), " out of: ", len(g))
            for e in dupes:
                g.remove_node(e)
        
        list_edge_feats = [[w['weight'] for (e1, e2, w) in g.edges(data=True)] for g in graphs]
        means = [np.mean(edge_feats) for edge_feats in list_edge_feats]
        sds = [np.std(edge_feats) for edge_feats in list_edge_feats]
        
        for g, mu, s in zip(graphs, means, sds):
            outlier = 0
            bad_edges = []
            for (e1, e2, w) in g.edges(data=True):
                if w['weight'] > mu + 4 * s:
                    bad_edges.append((e1, e2))
                    outlier += 1 
                        
            print("Num outliers: ", outlier, "out of: ", len(g.edges()))
            g.remove_edges_from(bad_edges)
            null_nodes = [n for n in g.nodes() if g.degree(n) == 0]
            print("Null nodes: ", len(null_nodes), "out of: ", len(g))
            g.remove_nodes_from(null_nodes)
        
        list_edge_feats = [[w['weight'] for (e1, e2, w) in g.edges(data=True)] for g in graphs]
        max_len = max([max(edge_feats) for edge_feats in list_edge_feats])
        print("Max edge length: ", max_len)
        
        for g in graphs:
            for (e1, e2, w) in g.edges(data=True):
                g[e1][e2]['weight'] = w['weight'] / max_len

else:
    graphs = create_training_graphs(cmd_args)

list_num_nodes = [len(g) for g in graphs]
list_num_edges = [len(g.edges()) for g in graphs]

print("Avg num nodes: ", np.mean(list_num_nodes))
print("Max num nodes: ", max(list_num_nodes))
print("Avg num edges: ", np.mean(list_num_edges))
print("Max num edges: ", max(list_num_edges))

## Updating w/ train valid
random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
print(cmd_args.g_type)    

source = (cmd_args.source if cmd_args.source > -1 else None)
num_graphs = len(graphs)
num_train = int(float(num_graphs) * cmd_args.train_ratio)
num_dev = num_train + int(float(num_graphs) * cmd_args.dev_ratio)
num_test_gt = num_graphs - num_dev
assert num_test_gt > 0

npr = np.random.RandomState(cmd_args.seed)
npr.shuffle(graphs)
graph_splits = {}
for phase, g_list in zip(['train', 'val', 'test'], [graphs[:num_train], graphs[num_train:num_dev], graphs[num_dev:]]):
    with open(os.path.join(cmd_args.save_dir, '%s-graphs.pkl' % phase), 'wb') as f:
        cur = []
        for g in tqdm(g_list):            
            cano_g = get_graph_data(g, cmd_args.node_order, cmd_args.leaf_order, order_only = False, global_source = source)                
            cur += cano_g
        cp.dump(cur, f, cp.HIGHEST_PROTOCOL)
    print('num', phase, len(g_list))
    graph_splits[phase] = g_list










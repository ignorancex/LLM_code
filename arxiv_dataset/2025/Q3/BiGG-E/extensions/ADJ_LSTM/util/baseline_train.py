import networkx as nx
import numpy as np
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import os
import sys
from extensions.common.configs import cmd_args

### Basline ER Model Code is adapted from GRAPHRNN
### Source: https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/baselines/baseline_simple.py

def get_number_of_nodes(graphs):
    nodes = []
    for graph in graphs:
        graph = graph.flatten()
        graph = graph[graph > -1]
        adj_len = len(graph)
        num_nodes = int(0.5 + (2 * adj_len + 0.25)**0.5)
        nodes.append(num_nodes)
    return np.array(nodes)
        

def get_number_of_edges(graphs):
    list_num_edges = []
    for graph in graphs:
        graph = graph.flatten()
        num_edges = torch.sum(graph > 0).item()
        list_num_edges.append(num_edges)
    return np.array(list_num_edges)

# def Graph_generator_baseline_train_rulebased(graphs,generator='Gnp'):
#     graph_nodes = get_number_of_nodes(graphs)
#     graph_edges = get_number_of_edges(graphs)
#     parameter = {}
#     for i in range(len(graph_nodes)):
#         nodes = graph_nodes[i]
#         edges = graph_edges[i]
#         # based on rule, calculate optimal parameter
#         if generator=='BA':
#             # BA optimal: nodes = n; edges = (n-m)*m
#             n = nodes
#             m = (n - np.sqrt(n**2-4*edges))/2
#             parameter_temp = [n,m,1]
#         if generator=='Gnp':
#             # Gnp optimal: nodes = n; edges = ((n-1)*n/2)*p
#             n = nodes
#             p = float(edges)/((n-1)*n/2)
#             parameter_temp = [n,p,1]
#         # update parameter list
#         if nodes not in parameter.keys():
#             parameter[nodes] = parameter_temp
#         else:
#             count = parameter[nodes][-1]
#             parameter[nodes] = [(parameter[nodes][i]*count+parameter_temp[i])/(count+1) for i in range(len(parameter[nodes]))]
#             parameter[nodes][-1] = count+1
#     # print(parameter)
#     return parameter

def Graph_generator_baseline_train_MLE(graphs):
    if torch.is_tensor(graphs):
        graph_nodes = get_number_of_nodes(graphs)
        graph_edges = get_number_of_edges(graphs)
    
    else:
        graph_nodes = np.array([len(g.nodes()) for g in graphs])
        graph_edges = np.array([len(g.edges()) for g in graphs])
    
    m = np.sum(graph_edges)
    n = np.sum(0.5 * graph_nodes * (graph_nodes - 1))
    
    p_hat = m / n
    print("Estimated Proportion: ", p_hat)
    return p_hat

def Graph_generator_baseline(ordered_train_graphs, weighted, generator, pred_num):
    graph_pred = []
    
    #parameter = Graph_generator_baseline_train_rulebased(ordered_train_graphs, generator)
    p_hat = Graph_generator_baseline_train_MLE(ordered_train_graphs)
    
    if torch.is_tensor(ordered_train_graphs):
        nodes = get_number_of_nodes(ordered_train_graphs)
    
    else:
        nodes = [len(g.nodes()) for g in ordered_train_graphs]
    
    #nodes = get_number_of_nodes(ordered_train_graphs)
    
    # keys_ = [key for key in parameter.keys()]
#     nodes = []
#     for key in keys_:
#         num_occur = parameter[key][2]
#         nodes = nodes + [key] * num_occur
#     
#     N = len(nodes)
    
    if weighted:
        if torch.is_tensor(ordered_train_graphs):
            ordered_train_graphs = ordered_train_graphs.flatten()
            weights = ordered_train_graphs[ordered_train_graphs > 0]
            W = weights.shape[0]
        
        else:
            weights = []
            for g in ordered_train_graphs:
                edges = sorted(g.edges(data=True), key=lambda x: x[0] * len(g) + x[1])
                g_weights = [x[2]['weight'] for x in edges]
                weights += g_weights
            W = len(weights)
    for i in range(pred_num):
        if num_nodes_list is not None:
            n = num_nodes_list[i]
        else:
            n = np.random.choice(nodes)
        #n = nodes[idx]
        g = nx.fast_gnp_random_graph(n, p_hat)
        if weighted:
            for (n1, n2) in g.edges():
                idx = np.random.choice(W)
                wt = weights[idx].item()
                g[n1][n2]['weight'] = wt
        graph_pred.append(g)
    return graph_pred
    

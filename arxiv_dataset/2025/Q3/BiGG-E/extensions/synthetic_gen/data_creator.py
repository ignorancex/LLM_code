import networkx as nx
import numpy as np
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
import os
import sys
import scipy
import scipy.stats
from collections import deque

# --------------------------------------------------------------
# Function `graph_load_batch` adapted from GRAN:
# https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/data_helper.py#L83
# Only this function is reused; the rest of the file is original.
# Copyright (c) 2019 Conan
# Licensed under the MIT License
# --------------------------------------------------------------

def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True,
                     edge_attributes=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = data_dir
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_coordinates.txt'.format(name)),
        delimiter=',')
  if edge_attributes:
    data_edge_att = np.loadtxt(
        os.path.join(path, '{}_edge_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  #edge_att_tuple = list(map(tuple, data_edge_att))
  weighted_edges = []

  for x, w in zip(data_tuple, data_edge_att):
      n1, n2 = x
      weight = w[0]
      weighted_edges.append((n1, n2, weight))

  G.add_weighted_edges_from(weighted_edges)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs

#######################################################################################
### SECTION ZERO: CREATE TRAINING GRAPHS AND APPLY ORDER ##############################
#######################################################################################

def create_training_graphs(args):
    g_type = args.g_type
    print("Generating graphs of type: ", g_type) 
    if g_type == "tree":
        graphs = graph_generator(args)
    elif g_type == "grid":
        graphs = get_rand_grid(args)
    elif g_type == "lobster": 
        graphs = get_rand_lobster(args)
    elif g_type == "er":
        graphs = get_rand_er(args)
    elif g_type == "franken":
        graphs = []
        args.num_graphs = 200
        graphs += get_rand_lobster(args)
        graphs += graph_generator(args)
        graphs += get_rand_er(args)
    elif g_type == "joint":
        graphs = joint_graph_generator(args)
    else:
        print("Graph type not yet implemented.")
        graphs = []
    return graphs

#######################################################################################
### SECTION ONE: CREATE TREE GRAPHS ###################################################
#######################################################################################


### Tree Generating Functions
def tree_generator(l, py_random, max_depth=99999):
    """
    Generates a random bifurcating tree with l leaves.
    More efficient than scanning the whole node list each time.
    """
    g = nx.Graph()

    # Start with a root node connected to two leaves
    g.add_edges_from([(0, 1), (0, 2)])
    next_node = 3
    active_leaves = [1, 2]
    depths = [1, 1]
    
    if max_depth > -1:
        assert max_depth > np.log2(l) + 1

    for _ in range(l - 2):
        # Pick a random active leaf
        idx = py_random.randrange(len(active_leaves))
        selected = active_leaves[idx]
        d = depths[idx]
        
        del active_leaves[idx]
        del depths[idx]
        
        # Add two new leaves
        left = next_node
        right = next_node + 1
        next_node += 2

        g.add_edges_from([(selected, left), (selected, right)])
        
        # Add new leaves if within max depth
        
        if d + 1 < max_depth:
            active_leaves.extend([left, right])
            depths.extend([d+1, d+1])

    return g


def graph_generator(args):
    '''
    Generates requested number of bifurcating trees
    Args:
        n: number of leaves
        num_graphs: number of requested graphs
    '''
    l = args.num_leaves
    num_graphs = args.num_graphs
    graphs = []
    npr = np.random.RandomState(args.seed)
    py_random = random.Random(args.seed)

    for _ in range(num_graphs):
        g = tree_generator(l, py_random)
        mu = npr.uniform(7, 13)
        weights = npr.gamma(mu * mu, 1 / mu, g.number_of_edges())

        # Add weights directly to edges
        for ((n1, n2), w) in zip(g.edges(), weights):
            g[n1][n2]['weight'] = w

        graphs.append(g)

    return graphs
    

# def joint_tree_constructor(T = 2, L = 1.5, eps = 0.1):
#     G = nx.Graph([(0, 1), (0, 2)])
#     path_lengths = {1: 0.0, 2: 0.0}
#     child_queue = 3
#     while len(path_lengths):
#         #print(G.edges(data = True))
#         cur_children = [child for child in path_lengths.keys()]
#         for child in cur_children:
#             parent = np.max([n for n in G.neighbors(child)])
#             t_i = 0.0
#             while t_i < 0.01:
#                 t_i = np.random.exponential(scale=L)
#             if path_lengths[child] + t_i > T - 2 * eps:
#                 t_i = T - path_lengths[child]
#                 min_eps = max(-eps, t_i - eps)
#                 u = np.random.uniform(-0.1, 0.1)
#                 t_i = T - path_lengths[child] + u
#             
#             
#             G[parent][child]['weight'] = t_i
#             path_lengths[child] += t_i
#             
#             if path_lengths[child] >= T - 2 * eps:
#                 # Stop splitting — this becomes a leaf
#                 del path_lengths[child]
#             
#             else:
#                 G.add_edges_from([(child, child_queue), (child, child_queue + 1)])
#                 path_lengths[child_queue] = path_lengths[child]
#                 path_lengths[child_queue + 1] = path_lengths[child]
#                 del path_lengths[child]
#                 child_queue += 2
#     return G
# 
# def joint_graph_constructor(N = 100, diff = 0.3, w0 = 0.5):
#     w = w0
#     g = nx.Graph()
#     for i in range(N):
#         for j in range(i):
#             low = max(0, w - diff)
#             high = min(1, w + diff)
#             w = np.random.uniform(low, high)
#             p = min(w, 1 - w)
#             e = int(p > np.random.uniform(0, 1))
#             if e == 1:
#                 g.add_weighted_edges_from([(i, j, w)])
#             else:
#                 w = w0
#     return g
# 
# def joint_graph_generator(args):
#     '''
#     Generates requested number of bifurcating trees
#     Args:
#     	n: number of leaves
#     	num_graphs: number of requested graphs
#     	constant_topology: if True, all graphs are topologically identical
#     	constant_weights: if True, all weights across all graphs are identical
#     	mu_weight: mean weight 
#     	scale: SD of weights
#     '''
#     n = args.num_leaves
#     num_graphs = args.num_graphs
#     graphs = []
#     
#     for _ in range(num_graphs):
#         g = nx.Graph()
#         while len(g) < 10:
#             g = joint_tree_constructor(2)
#         graphs += [g]
#     return graphs


## Joint Trees

def joint_graph_generator(args):
    num_graphs = args.num_graphs
    graphs = []
    py_random = random.Random(args.seed)
    
    for _ in range(num_graphs):
        g = generate_joint_graph(py_random)
        graphs += [g]
    return graphs

def generate_joint_graph(py_random, edge_min=1.0, edge_max=1.5, min_path_length=4.0): #min_path_length=3.0, edge_min=0.5, edge_max=2.5, seed=None):
    ## Test run -- will make these proper arguments
    #min_path_length=4.0
    edge_min=0.5
    edge_max=1.5
    
    G = nx.Graph()
    G.add_node(0)

    path_lengths = {0: 0.0}  # total path length from root (node 0) to each node
    next_node_id = 1
    queue = [0]

    while queue:
        parent = queue.pop(0)
        curr_path_len = path_lengths[parent]

        # If already over threshold, don't expand this node
        if curr_path_len > min_path_length:
            continue

        local_total = 0.0

        while True:
            weight = py_random.uniform(edge_min, edge_max)
            child = next_node_id
            next_node_id += 1

            new_path_len = curr_path_len + local_total + weight

            G.add_edge(parent, child, weight=weight)
            path_lengths[child] = curr_path_len + weight
            queue.append(child)

            local_total += weight

            if new_path_len > min_path_length:
                break  # stop once the *next* node passes the threshold
    return G



#######################################################################################
########## SECTION TWO: CREATE LOBSTER GRAPHS #########################################
#######################################################################################


def get_rand_lobster(args):
    npr = np.random.RandomState(args.seed)
    num_nodes = args.num_lobster_nodes
    p1 = args.p1
    p2 = args.p2
    min_nodes = args.min_nodes
    max_nodes = args.max_nodes
    num_graphs = args.num_graphs

    graphs = []

    for _ in range(num_graphs):
        # Keep sampling until graph is within size range
        while True:
            g = nx.random_lobster(num_nodes, p1, p2, seed=npr.randint(1e9))
            if min_nodes <= len(g) <= max_nodes:
                break

        for (n1, n2) in g.edges():
            w = scipy.stats.beta.rvs(5, 15, random_state=npr)
            g[n1][n2]['weight'] = w

        graphs.append(g)

    return graphs

#######################################################################################
############# SECTION THREE: CREATE ER GRAPHS #########################################
#######################################################################################


def get_rand_er(args):
    npr = np.random.RandomState(args.seed)
    min_er_nodes = args.min_er_nodes
    max_er_nodes = args.max_er_nodes
    p = args.p_er
    num_graphs = args.num_graphs

    graphs = []

    for _ in range(num_graphs):
        num_nodes = npr.randint(min_er_nodes, max_er_nodes + 1)
        g = nx.fast_gnp_random_graph(num_nodes, p, seed=npr)

        for n1, n2 in g.edges():
            z = scipy.stats.norm.rvs(random_state=npr)
            w = np.log(np.exp(z) + 1)  # softplus
            g[n1][n2]['weight'] = w

        graphs.append(g)

    return graphs



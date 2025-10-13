import os
import sys
import numpy as np
import random
import torch
import torch.optim as optim
import networkx as nx
import heapq
from collections import deque

# --------------------------------------------------------------
# Adapted from BiGG (Google Research):
# https://github.com/google-research/google-research/blob/c097eb6c850370c850eb7a90ab8a22fd2e1c730a/ugsl/input_layer.py#L103
# Functions adapted: get_node_map, apply_order, get_graph_data
# Copyright (c) Google LLC
# Licensed under the Apache License 2.0
# --------------------------------------------------------------

def canonical_dfs_order(G, root):
    visited = set()
    order = []
    def dfs(u):
        visited.add(u)
        order.append(u)
        for v in sorted(G.neighbors(u)):  # ensures deterministic traversal
            if v not in visited:
                dfs(v)
    dfs(root)
    return order

def canonical_bfs_order(G, root):
    visited = set([root])
    order = []
    queue = deque([root])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(G.neighbors(u)):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return order


# def weighted_bfs_tree(G, source):
#     visited = set()
#     queue = [(0, source)]
#     bfs_order = []
#     
#     while queue:
#         dist, node = heapq.heappop(queue)
#         if node in visited:
#             continue
#         visited.add(node)
#         bfs_order.append(node)
#         
#         for neighbor in G.neighbors(node):
#             if neighbor not in visited:
#                 weight = G[node][neighbor].get('weight', 1.0)
#                 heapq.heappush(queue, (dist + weight, neighbor))
#     
#     return bfs_order

def weighted_dfs_tree(G, source):
    visited = set()
    stack = [(0, source)]
    dfs_order = []
    
    while stack:
        _, node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        dfs_order.append(node)
        
        neighbors = []
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                weight = G[node][neighbor].get('weight', 1.0)
                neighbors.append((weight, neighbor))
        # Sort descending to prioritize heavier edges first
        neighbors.sort(reverse=False)
        stack.extend(neighbors)
    
    return dfs_order


def get_node_map(nodelist, shift=0):
    node_map = {}
    for i, x in enumerate(nodelist):
        node_map[x + shift] = i + shift
    return node_map


def apply_order(G, nodelist, order_only):
    if order_only:
        return nodelist
    node_map = get_node_map(nodelist)
    g = nx.relabel_nodes(G, node_map)
    return g

def order_tree(G, leaf_order = None): 
    n = len(G)
    leaves = sorted([x for x in G.nodes() if G.degree(x)==1])
    nodes = sorted([x for x in G.nodes() if x not in leaves])
    
    npl_dict, _ = nx.single_source_dijkstra(G, 0) 
    npl_list = [k for k in npl_dict.keys()]
    
    if leaf_order != "default":
        npl_n = [node for node in npl_list if node in nodes]
        npl_l = [node for node in npl_list if node in leaves]
        if leaf_order == "last": 
            npl_list = npl_n + npl_l
        else:
            npl_list = npl_l + sorted(npl_n, reverse=True) #Time Reversed...
    
    reorder = {}
    for k in range(n):
        reorder[npl_list[k]] = k
    new_G = nx.relabel_nodes(G, mapping = reorder)
    return new_G




def get_graph_data(G, node_order, leaf_order = None, order_only=False, global_source=None):
    out_list = []
    orig_node_labels = sorted(list(G.nodes()))
    orig_map = {x: i for i, x in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, orig_map)

    if node_order == 'default':
        out_list.append(apply_order(G, list(range(len(G))), order_only))
        
    elif node_order in ['DFS', 'BFS', 'WeightedDFS', 'WeightedBFS']:
        ### BFS & DFS from largest-degree node
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        
        # rank connected componets from large to small size
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        
        if global_source is not None:
            assert len(CGs) == 1
        
        node_list = []
        
        for cg in CGs:
            node_degree_list = [(n, d) for n, d in cg.degree()]
            degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
            
            source = (global_source if global_source is not None else degree_sequence[0][0])
            
            if node_order == 'BFS':
                if global_source is not None:
                    nodes = canonical_bfs_order(cg, root=source)
                
                else:
                    bfs_tree = nx.bfs_tree(cg, source=source)
                    nodes = list(bfs_tree.nodes())
                
                
            elif node_order == 'DFS':
                if global_source is not None:
                    nodes = canonical_dfs_order(cg, root=source)
                
                else:
                    dfs_tree = nx.dfs_tree(cg, source=source)
                    nodes = list(dfs_tree.nodes())
                
            elif node_order == 'WeightedDFS':
                nodes = weighted_dfs_tree(cg, source=source)
                
            node_list.extend(nodes)
        
        out_list.append(apply_order(G, node_list, order_only))
    
    else:
        print(f"[Warning] Node order '{node_order}' is not recognized. Defaulting to identity ordering.")
        out_list.append(apply_order(G, list(range(len(G))), order_only))
    
    return out_list
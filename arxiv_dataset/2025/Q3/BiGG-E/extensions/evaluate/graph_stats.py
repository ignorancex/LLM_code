from scipy.stats.distributions import chi2
import networkx as nx
import numpy as np
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import os
import scipy
from extensions.evaluate.mmd import *
from extensions.evaluate.mmd_stats import *
from extensions.common.configs import cmd_args, set_device

## Tuning Sigma for Weight MMD
def tune_sigma(train_graphs, g_type):
    random.shuffle(train_graphs)
    n = len(train_graphs)
    tot = n // 0.7
    m = int(0.2 * n)
    
    train_A = train_graphs[0:m]
    train_B = train_graphs[m:2*m]
    
    if g_type == "tree":
        sigmas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    elif g_type == "joint":
        sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
    else:
        sigmas = []
    
    ##null
    mmd_wt = mmd_weights_only(train_A, train_B, gaussian_wasserstein)
    print("=================")
    print("Null")
    print("MMD: ", mmd_wt)
    print("=================")
    
    for sigma in sigmas:
        mmd_wt = mmd_weights_only(train_A, train_B, gaussian_wasserstein, sigma)
        print("Sigma: ", sigma)
        print("MMD: ", mmd_wt)
        print("===============")
    
    import sys
    sys.exit() 





## Topology Check Functions
	
def check_leaf_lengths(G):
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    root = [n for n in G.nodes if G.degree(n) == 2][0]
    lengths = []
    for leaf in leaves:
        path = nx.shortest_path(G, source=root, target=leaf)
        dist = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        lengths.append(dist)
    return lengths


def get_tree_lengths(g):
  leaves = [l for l in g.nodes() if g.degree(l) == 1]
  path_lengths = []
  for leaf in leaves:
      path_length = nx.dijkstra_path_length(g, 0, leaf)
      path_lengths.append(path_length)
  return path_lengths


def correct_tree_topology_check(graphs):
  correct = 0
  true_trees = []
  for g in graphs:
    if is_bifurcating_tree(g):
        correct += 1
        true_trees.append(g)
  return correct / len(graphs), true_trees


def correct_tree_topology_check_two(graphs):
    props = []
    
    for g in graphs:
        root = [0]
        leaves = [n for n in g.nodes() if g.degree(n) == 1]
        internal = [n for n in g.nodes() if g.degree(n) == 3]
        good_nodes = root + leaves + internal
        props.append(len(good_nodes) / len(g))
    
    avg_prop = np.mean(props)
    return avg_prop


def correct_lobster_topology_check(graphs):
  correct = 0
  true_lobsters = []
  for g in graphs:
      if is_lobster(g):
          correct += 1
          true_lobsters.append(g)
  return correct / len(graphs), true_lobsters


def is_bifurcating_tree(g):
    if nx.is_tree(g):
        leaves = [n for n in g.nodes() if g.degree(n) == 1]
        internal = [n for n in g.nodes() if g.degree(n) == 3]
        root = [n for n in g.nodes() if g.degree(n) == 2]
        if 2*len(leaves) - 1 == len(g) and len(leaves) == len(internal) + 2 and len(root) == 1 and len(leaves) + len(internal)+ len(root) == len(g):
            return True
    return False


def is_lobster(graph):
    if not nx.is_tree(graph):
        return False
    g = nx.Graph(graph.edges())
    leaves = [l for l in g.nodes() if g.degree(l) == 1]
    g.remove_nodes_from(leaves)
    big_n = [n for n in g.nodes() if g.degree(n) >= 3]
    
    for n in big_n:
        big_neighbors = [x for x in g.neighbors(n) if g.degree(x) >= 2]
        if len(big_neighbors) > 2:
     	    return False
    return True





## Lobster Weight Statistics

def lobster_weight_statistics(graphs):
    means = []
    vars_ = []
    
    for g in graphs:
        weights = []
        for (n1, n2) in g.edges():
            weights.append(g[n1][n2]['weight'])
        
        means.append(np.mean(weights))
        vars_.append(np.var(weights, ddof = 1))
        
    mu_lo = np.percentile(means, 2.5)
    mu_hi = np.percentile(means, 97.5)
    print("Mean Estimate", np.mean(means))
    print('Empirical Interval: ', ' (' + str(mu_lo) + ',' + str(mu_hi) + ')')   
    
    var_lo = np.percentile(vars_, 2.5)
    var_hi = np.percentile(vars_, 97.5)
    print("Var Estimate", np.mean(vars_))
    print('Empirical Interval: ', ' (' + str(var_lo) + ',' + str(var_hi) + ')')    

def group_lobster_edges(g):
    backbone, one_hop, two_hop = group_lobster_nodes(g)
    
    edge_1, edge_2, edge_3 = [], [], []
    #print("NEW GRAPH")
    #print(g.edges())
    
    for (n1, n2, w) in g.edges(data=True):
        w = w['weight']
        if n1 in backbone and n2 in backbone:
            #print("Backbone edge: ")
            #print(n1, n2, w)
            edge_1.append(w)
        
        elif n1 in two_hop or n2 in two_hop:
            edge_3.append(w)
        
        else:
            edge_2.append(w)
    
    return edge_1, edge_2, edge_3

def group_lobster_nodes(g):
    g_prime = nx.Graph(g.edges)
    leaves1 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
    g_prime.remove_nodes_from(leaves1)
    
    if len(g_prime.edges()) == 1:
        leaves2 = []
        
    else:
        leaves2 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
        g_prime.remove_nodes_from(leaves2)
    
    backbone = sorted(list(g_prime.nodes))
    one_hop = []
    two_hop = []
    
    for n in leaves2:
        neighbors = list(g.neighbors(n))
        k = min(neighbors)
        if k in backbone:
            one_hop += [n]
        else:
            two_hop += [n]
    
    for n in leaves1:
        neighbors = list(g.neighbors(n))
        k = min(neighbors)
        if k in backbone:
            one_hop += [n]
        else:
            two_hop += [n]
    
    return backbone, one_hop, two_hop

## Tree Weight Statistics

def tree_weight_statistics(graphs, transform = False, prop = 0.0):
  ## Returns summary statistics on weights for graphs
  weights = []
  tree_vars = []
  tree_means = []
  #first_wts = []

  for T in graphs:
    T_weights = []
    for (n1, n2, w) in T.edges(data = True):
      if transform:
        t = np.log(np.exp(w['weight']) - 1)
      else:
        t = w['weight']
      T_weights.append(t)
      weights.append(t)
    tree_vars.append(np.var(T_weights, ddof = 1))
    tree_means.append(np.mean(T_weights))
  K = len(graphs[0])
  xbar = np.mean(weights)
  s = np.std(weights, ddof = 1)
  n = len(weights)

  mu_lo = np.round(xbar - 1.96 * s / n**0.5, 3)
  mu_up = np.round(xbar + 1.96 * s / n**0.5, 3)

  s_lo = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.975, df = n-1))**0.5, 3)
  s_up = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.025, df = n-1))**0.5, 3)

  mean_tree_var = np.mean(tree_vars)
  tree_var_lo = mean_tree_var - 1.96 * np.std(tree_vars, ddof = 1) / len(tree_vars)**0.5
  tree_var_up = mean_tree_var + 1.96 * np.std(tree_vars, ddof = 1) / len(tree_vars)**0.5
  
  mu_t_lo = np.percentile(tree_means, 2.5)
  mu_t_hi = np.percentile(tree_means, 97.5)
  
  s_t_lo = np.percentile(tree_vars, 2.5)**0.5
  s_t_hi = np.percentile(tree_vars, 97.5)**0.5
  
  print("Num of trees")
  print(len(graphs))
  
  results = [xbar, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var**0.5, tree_var_lo**0.5, tree_var_up**0.5]
  results_rounded = np.round(results, 3)
  
  print("Mean Estimates")
  print(results_rounded[0])
  print('95% CI: ', ' (' + str(results_rounded[1]) + ',' + str(results_rounded[2]), ')')
  print('Empirical Interval: ', ' (' + str(mu_t_lo) + ',' + str(mu_t_hi) + ')')
  
  print("SD Estimates")
  print(results_rounded[3])
  print('95% CI: ', ' (' + str(results_rounded[4]) + ',' + str(results_rounded[5]), ')')  
  
  print("Within Tree Variability")
  print(results_rounded[6])
  print('95% CI: ', ' (' + str(results_rounded[7]) + ',' + str(results_rounded[8]), ')')
  print('Empirical Interval: ', ' (' + str(s_t_lo) + ',' + str(s_t_hi) + ')')
  
  tree_means = [np.mean([w['weight'] for _, _, w in T.edges(data=True)]) for T in graphs]
  tree_means_sd = np.std(tree_means, ddof=1)
  print("SD of Tree Means (mu_Ts):", tree_means_sd)
  
  joint = False
  if joint:
    lengths = []
    lbars = []
    _, graphs = correct_tree_topology_check(graphs)
    
    for g in graphs:
        #g_lengths = get_tree_lengths(g)
        g_lengths = check_leaf_lengths(g)
        lengths += g_lengths
        lbars.append(np.mean(g_lengths))
        
    
    jbar = np.mean(lengths)
    js = np.var(lengths, ddof = 1) ** 0.5
    jbar_lo = jbar - 1.96 * js / len(lengths)**0.5
    jbar_hi = jbar + 1.96 * js / len(lengths)**0.5
    
    print("Global Estimates")
    print("Mean length: ", jbar)
    print("SD length: ", js)
    print('95% CI: ', ' (' + str(jbar_lo) + ',' + str(jbar_hi), ')')
    
    jbar = np.mean(lbars)
    js = np.var(lbars, ddof = 1) ** 0.5
    jbar_lo = np.percentile(lbars, 2.5)
    jbar_hi = np.percentile(lbars, 97.5)
    print("Per Tree Estimates")
    print("Mean length: ", jbar)
    print("SD length: ", js)
    print('Empirical Interval: ', ' (' + str(jbar_lo) + ',' + str(jbar_hi) + ')')
  


## MMD Statistics 

def get_mmd_stats(out_graphs, test_graphs, sigma=-1):
    mmd_degree = degree_stats(out_graphs, test_graphs)
    print("MMD Test on Degree Stats: ", mmd_degree)
    
    mmd_spectral_unweighted = spectral_stats(out_graphs, test_graphs, False)
    print("MMD on Specta of L Normalized, Unweighted: ", mmd_spectral_unweighted)
    
    mmd_cluster = clustering_stats(out_graphs, test_graphs)
    print("MMD on Clustering Coefficient: ", mmd_cluster)
    
    has_edge_feats = True
    if has_edge_feats:
        mmd_sepctral_weighted = spectral_stats(out_graphs, test_graphs, True)
        print("MMD on Specta of L Normalized, Weighted: ", mmd_sepctral_weighted)
        
        mmd_weights = mmd_weights_only(out_graphs, test_graphs, gaussian_wasserstein, sigma)
        print("MMD on Weights Only: ", mmd_weights)
        
        mmd_degree_wt = mmd_weighted_degree(out_graphs, test_graphs, gaussian_wasserstein)
        print("MMD on Weighted Degree: ", mmd_degree_wt)
        
        mmd_cluster_wt = weighted_clustering_stats(out_graphs, test_graphs)
        print("MMD on Weighted Clustering Coefficient: ", mmd_cluster_wt)
    
    #mmd_orbit = motif_stats(out_graphs, test_graphs)
    mmd_orbit = orbit_stats_all(out_graphs, test_graphs)
    print("MMD on Orbit: ", mmd_orbit)
    

## Graph Stats Function
def get_graph_stats(out_graphs, test_graphs, graph_type, g_type=None):
    if graph_type == "tree":
        prop, _ = correct_tree_topology_check(out_graphs)
        print("==========================================")
        print("Proportion Correct Topology: ", prop)
        
        prop2 = correct_tree_topology_check_two(out_graphs)
        print("Alt Proportion Correct Topology: ", prop2)
        
        print("Weight stats of Model Generated Graphs")
        test_stats2 = tree_weight_statistics(out_graphs, prop = prop)
        print("==========================================")
        print("Weight stats on Ground Truth Graphs")
        test_stats3 = tree_weight_statistics(test_graphs, prop = prop)
        print("==========================================")
        
        if test_graphs is None:
            return prop
        
        print("==========================================")
        print("MMD Statistics Comparing Generated and Ground Truth Graphs")
        get_mmd_stats(out_graphs, test_graphs, sigma=2)
        print("==========================================")
        
    
    elif graph_type == "lobster":
        prop, true_lobs = correct_lobster_topology_check(out_graphs)
        print("==========================================")
        print("Proportion Correct Lobster Graphs: ", prop)
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        print("Weight Stats of Model Generated Graphs")
        lobster_weight_statistics(out_graphs)
        print("==========================================")
        print("Weight Stats of Ground Truth Graphs")
        print("==========================================")
        lobster_weight_statistics(test_graphs)
        
        print("==========================================")
        print("MMD Statistics Comparing Generated and Ground Truth Graphs")
        get_mmd_stats(out_graphs, test_graphs)
        print("==========================================")
    
    elif graph_type == "db":
        weights = []
        for g in out_graphs:
            for (n1, n2, w) in g.edges(data=True):
                weights.append(w['weight']) 
        print("Mean weight: ", np.mean(weights))
        print("SD Weight: ", np.std(weights, ddof = 1))
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        print("==========================================")
        print("Ground Truth Graph Stats")
        
        weights = []
        for g in test_graphs:
            for (n1, n2, w) in g.edges(data=True):
                weights.append(w['weight']) 
        print("Mean weight: ", np.mean(weights))
        print("SD Weight: ", np.std(weights, ddof = 1))
        
        num_nodes = []
        num_edges = []
        for lobster in test_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        print("==========================================")
        get_mmd_stats(out_graphs, test_graphs)
    
    elif graph_type == "er":
        probs = []
        weights = []
        for g in out_graphs:
            n = len(g)
            if n <= 1:
                continue
            m = len(g.edges())
            p = 2 * m / (n * (n - 1))
            probs.append(p)
            for (n1, n2, w) in g.edges(data=True):
                if w['weight'] > 1e-8:
                    w_sm = np.log(np.exp(w['weight']) - 1)
                    weights.append(w_sm)
                else:
                    print("Small weight: ", w['weight'])
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        p_lo = np.percentile(probs, 2.5)
        p_hi = np.percentile(probs, 97.5)
        p_mu = np.mean(probs)
        print("Mean prob of edge existence: ", p_mu)
        print("95% Credible Interval: ", "(", p_lo, ", ", p_hi, ")")
        
        print("Mean SM weight: ", np.mean(weights))
        print("SD SM Weight: ", np.std(weights, ddof = 1))
        
        weights = []
        for g in test_graphs:
            for (n1, n2, w) in g.edges(data=True):
                w_sm = np.log(np.exp(w['weight']) - 1)
                weights.append(w_sm)
        
        print("Mean Test weight: ", np.mean(weights))
        print("SD Test Weight: ", np.std(weights, ddof = 1))
        
        get_mmd_stats(out_graphs, test_graphs)
    
    elif graph_type == "scale_test":
        prop, _ = correct_tree_topology_check(out_graphs)
        print("==========================================")
        print("Proportion Correct Topology: ", prop)
        
        prop2 = correct_tree_topology_check_two(out_graphs)
        print("Alt Proportion Correct Topology: ", prop2)
        
        print("Weight stats on sampled graphs: ")
        wt_stats = tree_weight_statistics(out_graphs)
        
        print("Weight stats on test graphs: ")
        wt_stats = tree_weight_statistics(test_graphs)
        print("==========================================")
        
        if cmd_args.num_leaves <= 500:
            mmd_sepctral = spectral_stats(out_graphs, test_graphs, False, cmd_args.num_leaves)
            print("MMD on Specta of L Normalized, Unweighted: ", mmd_sepctral)
        
        mmd_sepctral_weighted = spectral_stats(out_graphs, test_graphs, True, cmd_args.num_leaves)
        print("MMD on Specta of L Normalized, Weighted: ", mmd_sepctral_weighted)
        
        mmd_degree = degree_stats(out_graphs, test_graphs)
        print("MMD Test on Degree Stats: ", mmd_degree)
        
        mmd_weights = mmd_weights_only(out_graphs, test_graphs, gaussian_wasserstein, sigma=2.0)
        print("MMD on Weights Only: ", mmd_weights)
        print("==========================================")
    
    elif graph_type == "joint":
        correct = 0
        for g in out_graphs:
            if nx.is_tree(g):
                correct += 1
        
        prop = correct / len(out_graphs)
        print("Number of trees: ", prop)
        
        get_mmd_stats(out_graphs, test_graphs, sigma=0.3)
    
    else:
        print("Graph Type not yet implemented")
    return 0














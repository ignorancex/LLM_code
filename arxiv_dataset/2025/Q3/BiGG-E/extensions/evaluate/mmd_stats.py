import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time
import sys
from scipy.linalg import eigvalsh
from extensions.evaluate.mmd import *
from scipy.spatial.distance import pdist
import numpy as np

PRINT_TIME = False

# --------------------------------------------------------------
# This file modifies the original GraphRNN MMD evaluation code to work with BiGG-E outputs.
#  https://github.com/JiaxuanYou/graph-generation/blob/master/eval/mmd.py
# Copyright (c) 2017 Jiaxuan You, Rex Ying
# Licensed under the MIT License
# --------------------------------------------------------------

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def add_tensor(x,y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x+y

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref),len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        #total = 0
        #for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        #print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv,
                               sigma=1.0/10)#, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
        '3path' : [1, 2],
        '4cycle' : [8],
}
COUNT_START_STR = 'orbit counts: \n'

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1
    
    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph):
    path = '../../../evaluate/orca/tmp.txt'
    f = open(path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    
    output = sp.check_output(['../../../evaluate/orca/orca', 'node', '4', path, 'std'])
    output = output.decode('utf8').strip()
    
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
          for node_cnts in output.strip('\n').split('\n')])
    
    try:
        os.remove(path)
    except OSError:
        pass
    
    return node_orbit_counts
    

def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []
    
    num_matches_ref = []
    num_matches_pred = []
    
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]
    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        
        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())
            
        #hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)
        
    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        
        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())
            
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)
    
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian,
            is_hist=False)
    #print('-------------------------')
    #print(np.sum(total_counts_ref) / len(total_counts_ref))
    #print('...')
    #print(np.sum(total_counts_pred) / len(total_counts_pred))
    #print('-------------------------')
    return mmd_dist

def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []
 
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    #print(len(total_counts_pred))
    #print(len(total_counts_pred))
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv,
            is_hist=False, sigma=30.0)

    #print('-------------------------')
    #print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    #print('...')
    #print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    #print('-------------------------')
    return mmd_dist







## FROM https://github.com/lrjconan/GRAN/blob/master/utils/eval_helper.py
def spectral_worker(G, weighted, num_leaves = 999):
  # eigs = nx.laplacian_spectrum(G)
  if not weighted:
    G2 = nx.Graph(G.edges())
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G2).todense())  
  else:
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense()) 
   
   
  bins = min(200, num_leaves)
  
  spectral_pmf, _ = np.histogram(eigs, bins=bins, range=(-1e-5, 2), density=False)
  spectral_pmf = spectral_pmf / spectral_pmf.sum()
  # from scipy import stats  
  # kernel = stats.gaussian_kde(eigs)I just
  # positions = np.arange(0.0, 2.0, 0.1)
  # spectral_density = kernel(positions)

  # import pdb; pdb.set_trace()
  return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, weighted, num_leaves = 999, is_parallel=False):
  ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_ref_list):
        sample_ref.append(spectral_density)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        sample_pred.append(spectral_density)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
    #     sample_ref.append(spectral_density)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
    #     sample_pred.append(spectral_density)
  else:
    for i in range(len(graph_ref_list)):
      spectral_temp = spectral_worker(graph_ref_list[i], weighted, num_leaves)
      sample_ref.append(spectral_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], weighted, num_leaves)
      sample_pred.append(spectral_temp)
  # print(len(sample_ref), len(sample_pred))
  
  sigma = 1.0 # (2 / min(200, num_leaves))**0.5 

  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
  mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=sigma)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist


def collect_weights(list_graphs):
    list_graph_weights = []
    max_wt = -np.inf
    for g in list_graphs:
        #print("Hi")
        #print(g)
        graph_weights = [g[n1][n2]['weight'] for (n1, n2) in g.edges()]
        #print(graph_weights)
        if len(graph_weights) > 0:
            list_graph_weights.append(graph_weights)
            max_wt_g = max(graph_weights)
            max_wt = max(max_wt_g, max_wt)
        else:
            print("WARNING: Empty graph")
    return list_graph_weights, max_wt

# def mmd_weights_only(sample_graphs, target_graphs, kernel, bins= -1, num_leaves=-1):
#   A = [len(sample_graphs[i]) for i in range(len(sample_graphs))]
#   B = [len(target_graphs[i]) for i in range(len(target_graphs))]
#   n = np.median(A + B)
#   sample_list, max_a = collect_weights(sample_graphs)
#   target_list, max_b = collect_weights(target_graphs)
#   #print(sample_list)
#   
#   sample_ref = []
#   sample_pred = []
#   
#   #max_a = max([max(a) for a in sample_list])
#   #max_b = max([max(b) for b in target_list])
#   max_ = max(max_a, max_b)
#   
#   if num_leaves > 0:
#       bins = min(int(num_leaves), 200)
#   
#   elif bins == -1:
#       bins = int(n)
#   
#   for i in range(len(sample_list)):
#       hist_temp, _ = np.histogram(sample_list[i], range = (-1e-5, max_), bins=bins, density=False)
#       hist_temp = hist_temp / hist_temp.sum()
#       sample_ref.append(hist_temp)
#   
#   for i in range(len(target_list)):
#       hist_temp, _ = np.histogram(target_list[i], range = (-1e-5, max_), bins=bins, density=False)
#       hist_temp = hist_temp / hist_temp.sum()
#       sample_pred.append(hist_temp)
#   
#   mmd_dist = compute_mmd(sample_ref, sample_pred, kernel, is_parallel=False)
#   return mmd_dist

# Build normalized histograms over fixed range [0, max_]

def get_histogram_bins(all_weights, bin_width=0.01, dynamic_bin=True):
    all_weights = [np.atleast_1d(w) for w in all_weights]
    all_concat_weights = np.concatenate(all_weights)
    max_w = np.max(all_concat_weights)
    min_w = 0.0
    if dynamic_bin:
        min_w = np.min(all_concat_weights)
        if min_w < 1:
            min_w = 0
        
        else:
            bin_width = (max_w - min_w) / 100

    # Define how many total bins we need
    total_bins = int(np.ceil((max_w-min_w) / bin_width))
    bin_edges = np.linspace(min_w, min_w + total_bins * bin_width, total_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers, bin_width


def mmd_weights_only(sample_graphs, target_graphs, kernel, sigma=-1):
    # Collect weights
    sample_list, max_a = collect_weights(sample_graphs)
    target_list, max_b = collect_weights(target_graphs)
        
    all_weights = np.concatenate(sample_list + target_list)
    bin_edges, bin_centers, bin_width = get_histogram_bins(all_weights)
    
    def build_hist(weights):
        hist, _ = np.histogram(weights, bins=bin_edges, density=False)
        return hist / hist.sum() if hist.sum() > 0 else hist
    
    sample_ref = [build_hist(w) for w in sample_list]
    sample_pred = [build_hist(w) for w in target_list]    
    if sigma == -1:
        sigma = bin_width ** 0.5
    
    def kernel_with_centers(x, y):
        return kernel(x, y, bin_centers, sigma = sigma)
    # Compute MMD
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel_with_centers, is_parallel=False)
    return mmd_dist
    
    






###
def compute_trimmed_mean_weight(all_edge_weights, lower_q=1, upper_q=99):
    """
    Compute trimmed mean of edge weights using actual values,
    not interpolated percentiles.
    """
    sorted_weights = np.sort(all_edge_weights)
    n = len(sorted_weights)

    # Get indices for 1st and 99th percentiles (no interpolation)
    lower_idx = int(np.floor(n * lower_q / 100))
    upper_idx = int(np.ceil(n * upper_q / 100)) 

    trimmed = sorted_weights[lower_idx : upper_idx]
    return np.mean(trimmed)

def collect_weighted_degrees(graphs):
    """
    Returns:
        degree_list: list of np.arrays of weighted degrees per graph
        max_val: max weighted degree seen (for scaling/debug, optional)
    """
    degree_list = []
    max_val = 0
    for g in graphs:
        degs = np.array([
            sum(data['weight'] for _, _, data in g.edges(n, data=True))
            for n in g.nodes
        ])
        degree_list.append(degs)
        if len(degs) > 0:
            max_val = max(max_val, np.max(degs))
    return degree_list, max_val



def mmd_weighted_degree(sample_graphs, target_graphs, kernel, bins=-1):
    # Collect weighted degrees
    sample_list, max_a = collect_weighted_degrees(sample_graphs)
    target_list, max_b = collect_weighted_degrees(target_graphs)
    
    all_weights = np.concatenate([list(nx.get_edge_attributes(g, 'weight').values()) for g in sample_graphs + target_graphs])
    
    avg_weight = compute_trimmed_mean_weight(all_weights, lower_q=1, upper_q=99)

    all_values = np.concatenate(sample_list + target_list)
    bin_edges, bin_centers, _ = get_histogram_bins(all_values, bin_width = avg_weight, dynamic_bin = False)
    
    def build_hist(vals):
        hist, _ = np.histogram(vals, bins=bin_edges, density=False)
        return hist / hist.sum() if hist.sum() > 0 else hist

    sample_hists = [build_hist(w) for w in sample_list]
    target_hists = [build_hist(w) for w in target_list]

    def kernel_with_centers(x, y):
        return kernel(x, y, bin_centers, sigma = avg_weight ** 0.5)

    return compute_mmd(sample_hists, target_hists, kernel_with_centers, is_parallel=False)


def weighted_clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G, weight='weight').values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def weighted_clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(weighted_clustering_worker, 
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(weighted_clustering_worker, 
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for G in graph_ref_list:
            clustering_coeffs_list = list(nx.clustering(G, weight='weight').values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for G in graph_pred_list_remove_empty:
            clustering_coeffs_list = list(nx.clustering(G, weight='weight').values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    def kernel(x, y):
        return gaussian_tv(x, y, sigma=1.0/10)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel, is_parallel=False)
    
    if PRINT_TIME:
        print('Time computing weighted clustering MMD: ', datetime.now() - prev)

    return mmd_dist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
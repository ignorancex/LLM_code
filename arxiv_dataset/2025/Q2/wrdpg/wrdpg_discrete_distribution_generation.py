'''
Created on Sep 1, 2023

@author: bernardo
'''

from graspologic.simulations import sbm
from tqdm import trange

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import wrdpg


plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True

#########2-SBM + discrete for all nodes ###########
N = 500
ratio = 0.7
n_communities = 2
n = [int(N * ratio), int(N * (1 - ratio))]

p_c1 = 0.7
p_c2 = 0.5
q = 0.3

P = np.array([[p_c1, q],
              [q, p_c2]])

# Discrete distribution with values 1, 2, ..., 10
min_val = 1
values_range = 10
values = np.arange(values_range) + min_val

# P(1)=1/2 and uniform for the rest
probs = (1 / (2 * (values_range - 1))) * np.ones_like(values)
probs[4] = 1 / 2

# Sample graph from base model
sbm_graph = sbm(n=n, p=P, wt=wrdpg.discrete_distribution, wtargs=dict(values=values, probs=probs))
sbm_nx = nx.from_numpy_array(sbm_graph, create_using=nx.Graph)

# Sompute shortest path
shortest_paths_lengths = [l for u in sbm_nx for l in nx.single_source_dijkstra_path_length(sbm_nx, u, weight='weight').values()]

# Compute moments
K = probs.shape[0] + 1
moments_seq = [np.sum(probs * np.power(values, k)) for k in np.arange(1, K)]

# Compute theoretical latent positions
X = np.zeros((N, n_communities, K))
X_sbm = np.zeros((n_communities, n_communities, K))
X[:,:, 0] = [1, 0]
X_sbm[:,:, 0] = [1, 0]

values_mat = np.tile([0] + list(values), (N, N, 1))

for idx, moment in enumerate(moments_seq):
    X_c1 = np.array([np.sqrt(p_c1 * moment), 0])
    X_c2 = np.array([q * np.sqrt(moment / p_c1), np.sqrt(moment * (p_c2 - np.square(q) / p_c1))])
    X[:n[0],:, idx + 1] = X_c1
    X[n[0]:,:, idx + 1] = X_c2
    X_sbm[0,:, idx + 1] = X_c1
    X_sbm[1,:, idx + 1] = X_c2

# Goodness-of-fit diagnostics
n_simulations = 100
degrees = []
avg_shortest_paths = []
all_shortest_paths_proportion = {}
max_path_length = np.amax(shortest_paths_lengths)
betweenness = []

edges_moments = wrdpg.moments_from_latent(X)
p_hats, vals = wrdpg.estimate_edges_probabilities(edges_moments, values=values_mat, directed=False)

for k in trange(n_simulations, desc="Running WRDPG generation"):
    discrete_sbm_graph = wrdpg.sample_edges(p_hats, vals=vals, directed=False, loops=False)
    G_discrete = nx.from_numpy_array(discrete_sbm_graph, create_using=nx.Graph)
    
    # Compute degree distribution
    degrees.extend([G_discrete.degree(node, weight='weight') for node in G_discrete.nodes()])
    
    # Compute betweenness for each node
    betweenness.extend(nx.betweenness_centrality(G_discrete, weight='weight', backend='parallel').values())
    
    # Compute shortest paths lengths for each dyad 
    avg_shortest_paths.append(nx.average_shortest_path_length(G_discrete, weight='weight'))
    shortest_paths_lengths = [l for u in G_discrete for l in nx.single_source_dijkstra_path_length(G_discrete, u, weight='weight').values()]
    
    if np.amax(shortest_paths_lengths) > max_path_length:
        max_path_length = int(np.amax(shortest_paths_lengths))
        
    bins = np.arange(-0.5, np.amax(shortest_paths_lengths) + 1.5)
    shortest_paths_proportion, _ = np.histogram(shortest_paths_lengths, bins=bins, density=True)
    
    all_shortest_paths_proportion[k] = shortest_paths_proportion
 
# Plot metrics

fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=3, layout='constrained')

# Plot degree distribution 
degrees_sbm = [sbm_nx.degree(node, weight='weight') for node in sbm_nx.nodes()]
degrees_dist_sbm, deg_bins = np.histogram(degrees_sbm, bins='auto')
deg_bins_midpoints = (deg_bins[1:] + deg_bins[:-1]) / 2
degrees_dist_sbm = degrees_dist_sbm / N

degrees_dist_simulations, _ = np.histogram(degrees, bins=deg_bins)
degrees_dist_simulations = degrees_dist_simulations / len(degrees)

axs[0].plot(deg_bins_midpoints, degrees_dist_sbm, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
axs[0].bar(deg_bins_midpoints, degrees_dist_simulations, width=np.diff(deg_bins) * 0.8, color='#ff7f0e')
axs[0].set_xlabel('Degree')
axs[0].set_ylabel('Proportion of nodes')

# Plot betweenness
betweenness_sbm = list(nx.betweenness_centrality(sbm_nx, weight='weight', backend='parallel').values())
sbm_betwenness_hist, betweenness_bins = np.histogram(betweenness_sbm, bins='auto')
betweenness_bins_midpoints = (betweenness_bins[1:] + betweenness_bins[:-1]) / 2
sbm_betwenness_hist = sbm_betwenness_hist / N

betweenness_simulations, _ = np.histogram(betweenness, bins=betweenness_bins)
betweenness_simulations = betweenness_simulations / len(betweenness)

axs[1].plot(betweenness_bins_midpoints, sbm_betwenness_hist, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
axs[1].bar(betweenness_bins_midpoints, betweenness_simulations, width=np.diff(betweenness_bins) * 0.8, color='#ff7f0e')
axs[1].set_xlabel('Betweenness centrality')
axs[1].set_ylabel('Proportion of nodes')

# Plot shortest paths lengths

shortest_paths = {}
for shortest_path_length in np.arange(max_path_length + 1):
    shortest_paths[shortest_path_length] = []
    for k in all_shortest_paths_proportion:
        if len(all_shortest_paths_proportion[k]) > shortest_path_length:
            shortest_paths[shortest_path_length].append(all_shortest_paths_proportion[k][shortest_path_length])

bins = np.arange(-0.5, max_path_length + 1.5)
shortest_paths_proportion_sbm, _ = np.histogram(shortest_paths_lengths, bins=bins, density=True)

axs[2].plot(np.arange(len(shortest_paths_proportion_sbm)) + 1, shortest_paths_proportion_sbm, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])   
axs[2].boxplot(shortest_paths.values(), tick_labels=shortest_paths.keys())
axs[2].set_xlabel('Minimum geodesic distance')
axs[2].set_ylabel('Proportion of dyads')

plt.show()


'''
Created on Sep 1, 2023

@author: bernardo
'''

from graspologic.simulations import sbm
from graspologic.utils import symmetrize
from utils.moments_utils import normal_moments, exponential_moments
from tqdm import trange

import matplotlib.pyplot as plt
import maxent
import networkx as nx
import numpy as np


plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True

######### Exponential for all nodes ###########
N = 500
ratio = 0.7
n = [int(N * ratio), int(N * (1 - ratio))]
n_communities = len(n)

P = np.ones((n_communities, n_communities))

# Normal dist parameters
normal_params = [{'loc':6, 'scale':1}, {'loc':1, 'scale':0.1}]

# Exponential distribution parameters
scale = 3

Omega_exp = [0, 20]
Omega_normal_1 = [1, 10]
Omega_normal_2 = [0, 2]

gen = np.random.default_rng()

# Compute moments
K = 4

# Exponential distribution moments
moments_exp = exponential_moments(scale, K)

# Normal distribution moments
moments_normals = [normal_moments(params['loc'], params['scale'], K) for params in normal_params]

# Sample graph
wt = np.array([[gen.normal, gen.normal],
               [gen.normal, gen.exponential]])

wtargs = np.array([[normal_params[0], normal_params[1]],
                   [normal_params[1], dict(scale=scale)]])

# Sample graph from base model
graph = sbm(n=n, p=P, wt=wt, wtargs=wtargs)
graph_nx = nx.from_numpy_array(graph, create_using=nx.Graph)

# Compute shortest paths lengths
shortest_paths_lengths = [l for u in graph_nx for l in nx.single_source_dijkstra_path_length(graph_nx, u, weight='weight').values()]

lambguess = gen.random(len(moments_exp))
# Reconstruct using gradient descent on dual space for exponential dist
moment_matrix = np.vstack((moments_exp, moments_normals[0], moments_normals[1]))
lambdas_gd_exp, lambdas_gd_normal1, lambdas_gd_normal2 = maxent.solve_batch_log_continuous(moment_matrix,
                                                                                           Omegas=(Omega_exp,
                                                                                                   Omega_normal_1,
                                                                                                   Omega_normal_2))

# Goodness-of-fit diagnostics

n_simulations = 100
degrees = []
avg_shortest_paths = []
all_shortest_paths_proportion = {}
max_path_length = int(np.amax(shortest_paths_lengths))
betweenness = []

exp_solution_rv = maxent.maximum_entropy_rv(lambdas=lambdas_gd_exp, momtype=1, a=Omega_exp[0], b=Omega_exp[1])
exp_solution_normal1 = maxent.maximum_entropy_rv(lambdas=lambdas_gd_normal1, momtype=1, a=Omega_normal_1[0], b=Omega_normal_1[1])
exp_solution_normal2 = maxent.maximum_entropy_rv(lambdas=lambdas_gd_normal2, momtype=1, a=Omega_normal_2[0], b=Omega_normal_2[1])

for k in trange(n_simulations, desc="Running WRDPG generation"):
    # Sample from estimated densities
    samples_exp = exp_solution_rv.rvs(size=int(n[1] * (n[1] + 1) / 2))
    samples_normal1 = exp_solution_normal1.rvs(size=int(n[0] * (n[0] + 1) / 2))
    samples_normal2 = exp_solution_normal2.rvs(size=int(n[0] * n[1]))
    
    triu_inds_1 = np.triu_indices(n[0])
    sampled_graph = np.zeros((N, N))
    sampled_graph[triu_inds_1] = samples_normal1
    sampled_graph[:n[0], n[0]:] = samples_normal2.reshape((n[0], n[1]))
    
    triu_inds_2 = np.triu_indices(n[1])
    triu_inds_2 = (triu_inds_2[0] + n[0], triu_inds_2[1] + n[0])
    sampled_graph[triu_inds_2] = samples_exp
    sampled_graph = symmetrize(sampled_graph, method='triu')
    np.fill_diagonal(sampled_graph, 0)

    # sampled_graph = wrdpg.sample_continuous_edges(edges_rvs, directed=False, loops=False)
    G_sampled = nx.from_numpy_array(sampled_graph, create_using=nx.Graph)
    
    # Compute degree distribution
    degrees.extend([G_sampled.degree(node, weight='weight') for node in G_sampled.nodes()])
    
    # Compute betweenness for each node
    betweenness.extend(nx.betweenness_centrality(G_sampled, weight='weight', backend='parallel').values())
    
    # Compute shortest paths lengths for each dyad 
    avg_shortest_paths.append(nx.average_shortest_path_length(G_sampled, weight='weight'))
    shortest_paths_lengths = [l for u in G_sampled for l in nx.single_source_dijkstra_path_length(G_sampled, u, weight='weight').values()]
    
    if np.amax(shortest_paths_lengths) > max_path_length:
        max_path_length = int(np.amax(shortest_paths_lengths))
        
    bins = np.arange(-0.5, np.amax(shortest_paths_lengths) + 1.5)
    shortest_paths_proportion, _ = np.histogram(shortest_paths_lengths, bins=bins, density=True)
    
    all_shortest_paths_proportion[k] = shortest_paths_proportion


fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, layout='constrained')

# Plot degree distribution 
degrees_graph = [graph_nx.degree(node, weight='weight') for node in graph_nx.nodes()]
degrees_dist_graph, deg_bins = np.histogram(degrees_graph, bins='auto')
deg_bins_midpoints = (deg_bins[1:] + deg_bins[:-1]) / 2
degrees_dist_graph = degrees_dist_graph / N

degrees_dist_simulations, _ = np.histogram(degrees, bins=deg_bins)
degrees_dist_simulations = degrees_dist_simulations / len(degrees)

axs[0].plot(deg_bins_midpoints, degrees_dist_graph, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
axs[0].bar(deg_bins_midpoints, degrees_dist_simulations, width=np.diff(deg_bins) * 0.8, color='#ff7f0e')
axs[0].set_xlabel('Degree')
axs[0].set_ylabel('Proportion of nodes')

# Plot betweenness
# betweenness_graph = list(nx.betweenness_centrality(graph_nx, weight='weight',backend='parallel').values())
# betwenness_graph_hist, betweenness_bins = np.histogram(betweenness_graph,bins='auto')
# betweenness_bins_midpoints = (betweenness_bins[1:]+betweenness_bins[:-1])/2
# betwenness_graph_hist = betwenness_graph_hist/N
#
# betweenness_simulations, _ = np.histogram(betweenness,bins=betweenness_bins)
# betweenness_simulations = betweenness_simulations/len(betweenness)
#
# axs[1].plot(betweenness_bins_midpoints,betwenness_graph_hist, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
# axs[1].bar(betweenness_bins_midpoints, betweenness_simulations, width=np.diff(betweenness_bins)*0.8, color='#ff7f0e')
# axs[1].set_xlabel('Betweenness centrality')
# axs[1].set_ylabel('Proportion of nodes')

# Plot shortest paths lengths

shortest_paths = {}
for shortest_path_length in np.arange(max_path_length + 1):
    shortest_paths[shortest_path_length] = []
    for k in all_shortest_paths_proportion:
        if len(all_shortest_paths_proportion[k]) > shortest_path_length:
            shortest_paths[shortest_path_length].append(all_shortest_paths_proportion[k][shortest_path_length])
    # shortest_paths.append(np.array([all_shortest_paths_proportion[k][shortest_path_length] for k in all_shortest_paths_proportion if len(all_shortest_paths_proportion[k] > shortest_path_length)]))
    
bins = np.arange(-0.5, max_path_length + 1.5)

shortest_paths_proportion_graph, _ = np.histogram(shortest_paths_lengths, bins=bins, density=True)

axs[1].plot(np.arange(len(shortest_paths_proportion_graph)) + 1, shortest_paths_proportion_graph, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])   
axs[1].boxplot(shortest_paths.values(), tick_labels=shortest_paths.keys())
axs[1].set_xlabel('Minimum geodesic distance')
axs[1].set_ylabel('Proportion of dyads')

plt.show()


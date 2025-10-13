'''
Created on Apr 25, 2024

@author: bernardo
'''

import os

from graspologic.embed import AdjacencySpectralEmbed
from graspologic.simulations.simulations import sample_edges
from graspologic.utils import largest_connected_component
from matplotlib import colors as mplcolors
from scipy.sparse.linalg import eigs
from sklearn import metrics
from tqdm import trange

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from utils.football_dataset_utils import load_football_dataset
import wrdpg


# import leidenalg as la
# import igraph as ig
def partition_to_community_map(partition):
    """
    Converts a graph partition into a list where each index represents a node 
    and its value represents the community it belongs to.
    
    :param partition: List of sets, where each set contains nodes belonging to the same community.
    :return: Dictionary mapping each node to its community label.
    """
    community_map = {}
    
    for community_id, community in enumerate(partition):
        for node in community:
            community_map[node] = community_id
    
    return community_map
    

plt.rcParams['lines.linewidth'] = 3
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['axes.grid'] = True

directed = False

football_data_dir = 'your/data/dir/'
football_data_filename = 'AllMatches.csv'
football_data_path = os.path.join(football_data_dir,football_data_filename)

initial_year=2010
final_year=2016

G = load_football_dataset(football_data_path, initial_year=initial_year, final_year=final_year)

G_lcc = largest_connected_component(G)
N = G_lcc.number_of_nodes()
nodelist = list(G_lcc.nodes())

adj_matrix = nx.to_numpy_array(G_lcc,weight='weight',nodelist=nodelist)   

# Compute shortest paths lengths for football graph
shortest_paths_lengths = [l for u in G_lcc for l in nx.single_source_bellman_ford_path_length(G_lcc, u, weight='weight').values()]

gamma_louvain=1.5

# Detect communities with Louvain
communities_partition_football = nx.community.louvain_communities(G_lcc, weight='weight', resolution=gamma_louvain)
n_communities_football = len(communities_partition_football)
modularity_football = nx.community.modularity(G_lcc, communities_partition_football, resolution=gamma_louvain) 

community_map_football = partition_to_community_map(communities_partition_football)
communities_football = [community_map_football[node] for node in nodelist]

# To use Leiden instead uncomment following lines
# G_ig = ig.Graph.from_networkx(G_lcc)
# partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=gamma_louvain)
# communities_football = partition.membership

# Draw in map
ccodes = pd.read_csv(os.path.join(football_data_dir,'countries_codes.csv'))
# Table with countries coordinates
xrange = (np.min(ccodes.Capitallongitude),np.max(ccodes.Capitallongitude))
yrange = (np.min(ccodes.Capitallatitude),np.max(ccodes.Capitallatitude))

# Correct table
ccodes.loc[ccodes.Capitallongitude<-170,'Capitallongitude']=ccodes.loc[ccodes.Capitallongitude<-170,'Capitallongitude'] + 170+189

# Mercator projection
img = plt.imread("./figures/world.jpg")
mapWidth    = img.shape[1]
mapHeight   = img.shape[0]

easting = -80
northing = 77
# get x value
ccodes['x'] = (ccodes.Capitallongitude+180)*(mapWidth/360) + easting
# convert from degrees to radians
latRad = np.radians(ccodes.Capitallatitude)
# get y value
mercN = np.log(np.tan(np.pi/4 + latRad/2))
    
ccodes['y'] = (mapHeight/2)-(mapWidth*mercN/(2*np.pi)) + northing

posix = ccodes.set_index('Countryname')[['x','y']]
posix['arrays'] = list(posix.values)
posfijas = posix['arrays'].to_dict()


fig_world,ax_world = plt.subplots(figsize=(18,11), nrows=1,ncols=1, layout='constrained')
ax_world.imshow(img, extent=[0,mapWidth,mapHeight,0])

# Colors from https://colorbrewer2.org/
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d']
cmap = mplcolors.ListedColormap(colors)

nx.draw_networkx_nodes(G,posfijas,node_color=communities_football, nodelist=nodelist, ax=ax_world,
                       node_size=100, alpha=0.95, cmap=cmap)

nx.draw_networkx_edges(G,posfijas,edge_color='grey', nodelist=nodelist, ax=ax_world, alpha=0.4, width=[weight for s,t,weight in G.edges.data('weight')])
ax_world.axis('off')


# Embed A^k
d = 8
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='truncated')


max_val = np.amax(adj_matrix)
values = np.arange(1,max_val+1)
n_moments = 3
values_mat = np.tile(values,(N,N,1))

psd_adj_matrix = wrdpg.make_psd(adj_matrix)

# Get connectivity pattern
unweighted_adj_matrix = (adj_matrix > 0)

X_0 = ase.fit_transform(unweighted_adj_matrix)
P = X_0@X_0.T
P[P<0] = 0
P[P>1] = 1

if directed:
    X_seq = np.empty((2,N,d,n_moments))
else:
    X_seq = np.empty((N,d,n_moments))

Q_seq = np.empty((d,d,n_moments))


for k in np.arange(n_moments):
    A = np.power(adj_matrix,k)
    Xhats = ase.fit_transform(A)
    if type(Xhats) is tuple:
        (Xl,Xr) = Xhats
    else:
        Xl=Xr=Xhats

    if directed:
        X_seq[0,:,:,k] = Xl
        X_seq[1,:,:,k] = Xr
    else:
        X_seq[:,:,k] = Xl

    # Compute the matrix Q
    (w,v) = eigs(A,k=d,which='LM')
    wabs = np.array(list(zip(-np.abs(w), -np.sign(np.real(w)))), dtype=[('abs', 'f4'), ('sign', 'i4')])
    w = w[np.argsort(wabs,order=['abs','sign'])]
    Q = np.diag(np.sign(np.real(w)))
    Q_seq[:,:,k] = Q



moments_mat = wrdpg.moments_from_latent(X_seq, Q_seq=Q_seq,directed=directed)
psd_moments_mat = wrdpg.check_psd_moments_mat(moments_mat)

# Estimated moments boxplots
fig_moments, ax_moments = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
moments_idxs = np.arange(n_moments)

medianprops = dict(linewidth=2.5, color='darkgreen')
boxprops = dict(linestyle='--', linewidth=2.5, color='darkgreen')
flierprops = dict(marker='o', markerfacecolor='None', markeredgecolor='darkgreen')
whiskerprops = dict(linewidth=2.5, color='darkgreen',alpha=0.8)
capprops = whiskerprops

ax_moments.boxplot(moments_mat.reshape((N**2,-1)),tick_labels=moments_idxs,
                   positions=moments_idxs, label='Moments from latent positions',
                   medianprops = dict(linewidth=2.5, color='maroon'), whis=(0,100))

ax_moments.boxplot(psd_moments_mat.reshape((N**2,-1)),tick_labels=moments_idxs,
                   positions=moments_idxs, label='PSD moments', medianprops=medianprops,
                   boxprops=boxprops, flierprops=flierprops, whis=(0,100), whiskerprops=whiskerprops,
                   capprops=capprops)

ax_moments.set_title('Estimated moments for football dataset')
ax_moments.set_yscale('log')

ax_moments.set_ylabel('Moment value')
ax_moments.set_xlabel('Moment number')

# Estimate edges probabilities

p_hats, vals = wrdpg.estimate_edges_probabilities(psd_moments_mat, values=values_mat, directed=directed,
                                                  P_edges=P, reg_strength=1e-4, parallel=True,verbose=False)

fig_phats, ax_phats = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
probs_idxs = np.arange(max_val+1)
ax_phats.boxplot(np.array(p_hats),tick_labels=[f'$p_{{{i}}}$' for i in [0]+list(values)],
                     positions=probs_idxs, label='Probabilities from latent positions',
                     medianprops = dict(linewidth=2.5, color='maroon'), whis=(0,100))


fig_phats.suptitle('Estimated link probabilities')



# Goodness-of-fit diagnostics

n_simulations = 100
degrees = []
all_shortest_paths_proportion = {}
max_path_length = int(np.amax(shortest_paths_lengths))
betweenness = []
lcc_sizes = []
n_communities_list = []
modularities = []
score_funcs = [
    ("V-measure", metrics.v_measure_score),
    ("ARI", metrics.adjusted_rand_score),
    ("AMI", metrics.adjusted_mutual_info_score),
]

scores = {score:[] for score,_ in score_funcs}


for k in trange(n_simulations, desc="Running WRDPG generation"):
    discrete_graph = wrdpg.sample_edges(p_hats, vals=vals,directed=directed, parallel=True)
    if directed:
        G_discrete = nx.from_numpy_array(discrete_graph, create_using=nx.DiGraph)
    else:
        G_discrete = nx.from_numpy_array(discrete_graph, create_using=nx.Graph)
    
    G_discrete = nx.relabel_nodes(G_discrete, {i:country for i,country in enumerate(nodelist)}, copy=False)
    
    G_discrete_lcc = largest_connected_component(G_discrete)
    lcc_sizes.append(G_discrete_lcc.number_of_nodes())
    
    # Compute degree distribution
    degrees.extend([G_discrete_lcc.degree(node, weight='weight') for node in G_discrete_lcc.nodes()])
    
    # Compute betweenness for each node
    betweenness.extend(nx.betweenness_centrality(G_discrete_lcc, weight='weight', backend="parallel").values())
    
    #Compute shortest paths lengths for each dyad 
    shortest_paths_lengths = [l for u in G_discrete_lcc for l in nx.single_source_bellman_ford_path_length(G_discrete_lcc, u, weight='weight').values()]
    
    if np.amax(shortest_paths_lengths) > max_path_length:
        max_path_length = int(np.amax(shortest_paths_lengths))
    
    bins = np.arange(-0.5,np.amax(shortest_paths_lengths)+1.5)
    shortest_paths_proportion,_ = np.histogram(shortest_paths_lengths,bins=bins, density=True)
    
    all_shortest_paths_proportion[k] = shortest_paths_proportion
    
    # Detect communitites
    communities_partition_synthetic = nx.community.louvain_communities(G_discrete, weight='weight', resolution=gamma_louvain)
    n_communities_list.append(len([comm for comm in communities_partition_synthetic if len(comm)>1]))
    modularities.append(nx.community.modularity(G_discrete, communities_partition_synthetic, resolution=gamma_louvain))
    
    community_map_synthetic = partition_to_community_map(communities_partition_synthetic)
    communities_synthetic = [community_map_synthetic[node] for node in nodelist]

    # To use Leiden instead uncomment following lines
    # G_discrete_ig = ig.Graph.from_networkx(G_discrete)
    # partition_synthetic = la.find_partition(G_discrete_ig, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=gamma_louvain)
    # communities_synthetic = partition_synthetic.membership
    
    # Compute clusters scores
    for score_name, func in score_funcs:
        scores[score_name].append(func(communities_football,communities_synthetic))

fig_hist, ax_hist = plt.subplots(figsize=(10,10),nrows=1, ncols=2)
ax_hist[0].hist(adj_matrix.flatten(),bins=np.arange(-0.5,int(np.amax(adj_matrix))+1.5),density=True)
ax_hist[0].set_title('Original')
ax_hist[1].hist(discrete_graph.flatten(),bins=np.arange(-0.5,int(np.amax(discrete_graph))+1.5),density=True)
ax_hist[1].set_title('WRDPG')
fig_hist.suptitle('Degree distribution')


fig, axs = plt.subplots(figsize=(20,10),nrows=1, ncols=3)

# Plot degree distribution 
degrees_football = [G_lcc.degree(node,weight='weight') for node in G_lcc.nodes()]
degrees_dist_football, deg_bins = np.histogram(degrees_football,bins='auto')
deg_bins_midpoints = (deg_bins[1:]+deg_bins[:-1])/2
degrees_dist_football = degrees_dist_football/N

degrees_dist_simulations, simulations_bins = np.histogram(degrees,bins='auto')
degrees_dist_simulations = degrees_dist_simulations/len(degrees)
simulations_bins_midpoints = (simulations_bins[1:]+simulations_bins[:-1])/2

degrees_dist_simulations, _ = np.histogram(degrees,bins=deg_bins)
degrees_dist_simulations = degrees_dist_simulations/len(degrees)


axs[0].plot(deg_bins_midpoints, degrees_dist_football, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
axs[0].bar(deg_bins_midpoints, degrees_dist_simulations, width=np.diff(deg_bins)*0.8, color='#ff7f0e')
axs[0].set_xlabel('Degree')
axs[0].set_ylabel('Proportion of nodes')

# Plot betweenness
betweenness_sbm = list(nx.betweenness_centrality(G_lcc, weight='weight', backend='parallel').values())
sbm_betwenness_hist, betweenness_bins = np.histogram(betweenness_sbm,bins=10)
betweenness_bins_midpoints = (betweenness_bins[1:]+betweenness_bins[:-1])/2
sbm_betwenness_hist = sbm_betwenness_hist/N

betweenness_simulations, _ = np.histogram(betweenness,bins=betweenness_bins)
betweenness_simulations = betweenness_simulations/len(betweenness)

axs[1].plot(betweenness_bins_midpoints,sbm_betwenness_hist, marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])
axs[1].bar(betweenness_bins_midpoints, betweenness_simulations, width=np.diff(betweenness_bins)*0.8, color='#ff7f0e')
axs[1].set_xlabel('Betweenness centrality')
axs[1].set_ylabel('Proportion of nodes')

# Plot shortest paths lengths
shortest_paths = {}
for shortest_path_length in np.arange(max_path_length+1):
    shortest_paths[shortest_path_length] = []
    for k in all_shortest_paths_proportion:
        if len(all_shortest_paths_proportion[k]) > shortest_path_length:
            shortest_paths[shortest_path_length].append(all_shortest_paths_proportion[k][shortest_path_length])


bins = np.arange(-0.5,max_path_length+1.5)

shortest_paths_proportion_sbm,_ = np.histogram(shortest_paths_lengths,bins=bins, density=True)

axs[2].plot(np.arange(len(shortest_paths_proportion_sbm)) + 1, shortest_paths_proportion_sbm,marker='o', markerfacecolor='none', markeredgewidth=plt.rcParams['lines.linewidth'])   
axs[2].boxplot(shortest_paths.values(), tick_labels=shortest_paths.keys())
axs[2].set_xlabel('Minimum geodesic distance')
axs[2].set_ylabel('Proportion of dyads')

fig.subplots_adjust(left=0.10,right=0.98,top=0.96,bottom=0.11,hspace=0.2,wspace=0.31)

fig_clustering, axs_clustering = plt.subplots(figsize=(20,10),nrows=1, ncols=2, layout='constrained')
axs_clustering[0].hist(n_communities_list, rwidth=0.8, color='#ff7f0e',density=True, bins = np.arange(np.amin(n_communities_list)-0.5,np.amax(n_communities_list)+1.5))
axs_clustering[0].axvline(n_communities_football, ls='--')
axs_clustering[0].set_xlabel('Number of communities in lcc')
axs_clustering[0].set_ylabel('Proportion of graphs')
axs_clustering[1].boxplot(scores.values(),  tick_labels=scores.keys())
axs_clustering[1].set_ylabel('Score value')


plt.show()

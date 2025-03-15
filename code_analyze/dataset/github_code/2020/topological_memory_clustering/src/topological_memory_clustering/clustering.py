from __future__ import print_function, division

import numpy as np
from scipy.cluster.vq import kmeans,vq
from time import time

import scipy
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet

from pylab import rcParams
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

__all__ = ['class_color',
           'get_colors_from_labels',
           'preprocess_dataset_for_clustering',
           'kmeans_clustering']

# class_color = ['orange', 'blue', 'red', 'green', 'magenta', 'purple']
class_color = ['tab:blue',
               'tab:orange',
               'tab:green',
               'tab:red',
               'tab:purple',
               'tab:brown',
               'tab:pink',
            #    'tab:gray',
               'tab:olive',
               'tab:cyan',
               'aquamarine',
               'cornflowerblue',
               'crimson',
               'teal',
               'palegreen',
               'lemonchiffon',
               'tomato',
               'indigo']

def get_colors_from_labels(labels):
    colors = [''] * labels.shape[0]
    for i in range(labels.shape[0]):
        if labels[i] >= len(class_color):
            print("CLASS OUT OF COLORS", i, labels[i])
            # colors[i] = 'white'
            colors[i] = 'tab:gray'
        else:
            colors[i] = class_color[labels[i]]
    return colors

def preprocess_dataset_for_clustering(samples_X):
    sample_dim = samples_X.shape[0]
    state_dim = samples_X.shape[1]
    time_dim = samples_X.shape[2]
    X = np.zeros((sample_dim,state_dim * time_dim))
    for sample_i in range(sample_dim):
        for coordinate_i in range(state_dim):
            X[sample_i,time_dim*coordinate_i:(coordinate_i+1)*time_dim] = samples_X[sample_i,coordinate_i,:]
    return X

def kmeans_clustering(samples_X, num_clusters, debug=False):
    X = preprocess_dataset_for_clustering(samples_X)

    kmeans_start = time()
    centroids,_ = kmeans(X, num_clusters)
    idx,_ = vq(X,centroids)
    kmeans_end = time()
    debug and print("k-means took", kmeans_end-kmeans_start)
    #print(centroids.shape, X[idx==0,:].shape)
    return centroids, idx

def compute_truncated_dendrogram(samples_X, linkage_method='ward', truncation=12, horizontal_lines=[], fig=None, debug=False):
    if fig is None:
        plt.figure()

    X = preprocess_dataset_for_clustering(samples_X)

    classical_start = time()
    Z = linkage(X, linkage_method)
    dendrogram(Z, truncate_mode='lastp', p=truncation, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
    classical_end = time()
    debug and print("Classical took", classical_end-classical_start)

    plt.title('Truncated Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')

    for line in horizontal_lines:
        plt.axhline(y=line[0], color=line[1], linestyle='--')

    return fig

def agglomerative_clustering(samples_X, num_clusters, linkage_method='complete', debug=False):
    # affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    X = preprocess_dataset_for_clustering(samples_X)

    ac_start = time()
    Hclustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method) 
    # affinity='precomputed' / Hclustering.fit(D)
    Hclustering.fit(X)
    ac_end = time()
    debug and print("Agglomerative clustering took", ac_end-ac_start)

    # Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    # Hclustering.fit(X) 
    # sm.accuracy_score(y, Hclustering.labels_)

    if debug:
        for i in range(num_clusters):
            print(i, np.sum(Hclustering.labels_==i))

    return Hclustering.labels_

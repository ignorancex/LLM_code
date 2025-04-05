#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: metric_op.py
@time: 2020/7/15 13:56
@desc:
'''
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from scipy import sparse as sp
import igraph as ig
from igraph import Graph as IG
import networkx as nx
import copy
# from utils.parallel_op import _parallel
from utils.load_op import clusterLabel_2_community


def cal_nmi(labels_true, labels_pred):
    """
    compute nmi
    :param labels_true:
    :param labels_pred:
    :return:
    """
    return metrics.normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)


def cal_ARI(labels_true, labels_pred):
    """
    compute ARI
    :param labels_true:
    :param labels_pred:
    :return:
    """
    return metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)


def cal_f1(labels_true, labels_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    n = len(labels_true)
    # a lookup table
    for i in range(n):
        for j in range(i + 1, n):
            same_label = (labels_true[i] == labels_true[j])
            same_cluster = (labels_pred[i] == labels_pred[j])
            if same_label:
                if same_cluster:
                    TP += 1
                else:
                    FN += 1
            elif same_cluster:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = 2 * precision * recall / (precision + recall)
    return fscore


def evaluate_results(input, community, labels_true, graphLoader='ig'):
    global Q
    labels_pred = [""] * len(labels_true)
    labels_pred_dict = {}
    # print(len(labels_true))
    for commID, comm in enumerate(community):
        for node in comm:
            labels_pred[node] = commID
            labels_pred_dict[node] = commID
    if type(input) == 'str':
        input = nx.read_gml(input, label='id')
        Q = modularity(labels_pred_dict, input)
    elif type(input) == IG:
        # print(community)
        Q = input.modularity(community)
    elif type(input) == nx.classes.graph.Graph:
        Q = modularity(labels_pred_dict, input)

    # cal nmi
    nmi = cal_nmi(labels_true, labels_pred)
    # cal ARI
    ARI = cal_ARI(labels_true, labels_pred)
    f1 = cal_f1(labels_true, labels_pred)
    return Q, nmi, ARI, f1


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    """
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        # com = partition[str(node)]
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            # if partition[str(neighbor)] == com:
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def cal_clusterConsensus(connect_components_labels, num_component, cooccurrence_matrix):
    weighted_score_list = []
    for commID in range(num_component):
        comm = np.where(connect_components_labels == commID)[0]
        commSize = len(comm)

        if commSize <= 1:
            intraCommScore = 0
        else:
            intraCommMatrix = cooccurrence_matrix[comm][:, comm]  # adj of community
            intraCommScore = intraCommMatrix.sum() / ((commSize - 1) * commSize / 2)

        weighted_score_list.append(commSize * intraCommScore)
    clusterConsensus = sum(weighted_score_list) / cooccurrence_matrix.shape[0]
    return clusterConsensus


def cal_clusterConsensus_1(cooccurrence_matrix, threshold, cooccurrence_matrix_dup):
    cooccurrence_matrix.data[cooccurrence_matrix.data < threshold] = 0
    cooccurrence_matrix.eliminate_zeros()  # Remove zero entries from the matrix
    num_component, connect_components_labels = sp.csgraph.connected_components(cooccurrence_matrix, directed=False)

    return (0, None) if num_component == 1 else (
        cal_clusterConsensus(connect_components_labels, num_component, cooccurrence_matrix_dup), connect_components_labels)


def cal_clusterConsensus_parallel_worker(x):
    return cal_clusterConsensus_1(*x)


def threshold_search(cooccurrence_matrix):
    global optimal_connected_components
    cooccurrence_matrix_dup = copy.deepcopy(cooccurrence_matrix)
    candidate_thresholds = sorted(set(cooccurrence_matrix.data))
    # print(candidate_thresholds)
    # print(cooccurrence_matrix_dup)
    optimal_threshold = -1
    optimal_score = np.finfo(np.float64).min

    for threshold in candidate_thresholds:
        cooccurrence_matrix.data[cooccurrence_matrix.data < threshold] = 0
        cooccurrence_matrix.eliminate_zeros()  # Remove zero entries from the matrix
        num_component, connect_components_labels = sp.csgraph.connected_components(cooccurrence_matrix, directed=False)

        if num_component == 1: continue
        partitionScore = cal_clusterConsensus(connect_components_labels, num_component, cooccurrence_matrix_dup)
        # print('!!!!!!!!!!!', partitionScore)
        if partitionScore > optimal_score:
            optimal_score = partitionScore
            optimal_threshold = threshold
            optimal_connected_components = ig.Clustering(connect_components_labels)
    try:
        return optimal_threshold, optimal_connected_components

    except:
        optimal_connected_components = ig.Clustering([0 for n in range(cooccurrence_matrix.shape[0])])
        optimal_threshold = 2
        return optimal_threshold, optimal_connected_components



def cal_averageSimilarity(core_components, simMatrix_list):
    all_similarty_assign = np.zeros([len(simMatrix_list), simMatrix_list[0].shape[0]])

    for i, similarity_matrix in enumerate(simMatrix_list):

        averageSimilarity_matrix = np.zeros([len(core_components), similarity_matrix.shape[0]])
        for commID, comm in enumerate(core_components):
            averageCommSimilarity = similarity_matrix[comm].astype(np.float16).mean(axis=0)
            averageSimilarity_matrix[commID, :] = averageCommSimilarity

        all_similarty_assign[i, :] = np.argmax(averageSimilarity_matrix, axis=0)

    # assign via vote
    maxCommID = list(map(np.argmax, map(np.bincount, all_similarty_assign.T.astype(np.int))))
    return maxCommID



def kmeans(X, num_cluster):
    num_node = X.shape[0]
    if num_node < 2000:
        km_cluster = KMeans(n_clusters=num_cluster, n_init=10)
    else:
        km_cluster= MiniBatchKMeans(n_clusters=num_cluster, batch_size=50, verbose=1)

    km_cluster.fit(X)
    labels_pred = km_cluster.labels_.tolist()

    return clusterLabel_2_community(labels_pred)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: communityDetection_op.py
@time: 2020/7/15 13:39
@desc:
'''
from karateclub import LabelPropagation, SCD, GEMSEC, EdMot

from functools import wraps
from utils.graph_op import read_igraph, igraph_to_sparse_matrix
from utils.metric_op import *

from utils.n2v_op import *


def add_labels(func):
    @wraps(func)
    def get_labels(labels_true, input, cal_target):
        # 获取社团检测结果
        comm_cluster = func(labels_true, input, cal_target)

        if cal_target == 'Q':
            return input.modularity(comm_cluster)
        else:
            labels_pred = [""] * len(labels_true)
            for commID, comm in enumerate(comm_cluster):
                for node in comm:
                    labels_pred[node] = commID
                    # partition[node] = commID

            if cal_target == 'nmi':
                return cal_nmi(labels_true=labels_true, labels_pred=labels_pred)
            elif cal_target == 'ARI':
                return cal_ARI(labels_true=labels_true, labels_pred=labels_pred)

    return get_labels


# 社团检测算法
@add_labels
def fastgreedy(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_fastgreedy(g)
    else:
        community = IG.community_fastgreedy(g, weights='weight')

    comm_cluster = community.as_clustering()
    return comm_cluster


@add_labels
def infomap(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input

    if not weight:
        community = IG.community_infomap(g)
    else:
        community = IG.community_infomap(g, edge_weights='weight')
    # print(community)
    comm_cluster = community
    return comm_cluster


@add_labels
def label_propagation(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_label_propagation(g)
    else:
        community = IG.community_label_propagation(g, weights='weight')
    # print(community)
    comm_cluster = community
    return comm_cluster


@add_labels
def walktrap(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_walktrap(g, steps=2)
    else:
        community = IG.community_walktrap(g, weights='weight', steps=2)
    # print(community)
    # print(community.as_clustering())
    comm_cluster = community.as_clustering()
    return comm_cluster


@add_labels
def leading_eigenvector(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_leading_eigenvector(g)
    else:
        community = IG.community_leading_eigenvector(g, weights='weight')
    # print(community)
    comm_cluster = community
    return comm_cluster


@add_labels
def multilevel(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_multilevel(g)
    else:
        community = IG.community_multilevel(g, weights='weight')
    # print(community)
    comm_cluster = community
    return comm_cluster


@add_labels
def edgebetweenness(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = read_igraph(input)
    else:
        g = input
    if not weight:
        community = IG.community_edge_betweenness(g)
    else:
        community = IG.community_edge_betweenness(g, weights='weight')
    # print(community)
    comm_cluster = community.as_clustering()
    return comm_cluster


def commDict_to_commCluster(commDict):
    commNum = len(set(commDict.values()))
    a_arr = np.array(list(commDict.items())).T
    commCluster = [''] * commNum
    for i in range(commNum):
        commCluster[i] = np.where(a_arr[1] == i)[0].tolist()
    return commCluster


@add_labels
def scd(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = nx.read_gml(input, label='id')
    elif type(input) == IG:
        g = nx.from_scipy_sparse_matrix(igraph_to_sparse_matrix(input))
    else:
        g = input
    model = SCD()
    model.fit(g)
    community_dict = model.get_memberships()

    clusterID_relabel = {old_id: i for i, old_id in enumerate(set(community_dict.values()))}
    community_dict = {k: clusterID_relabel[v] for k, v in community_dict.items()}
    return commDict_to_commCluster(community_dict)


@add_labels
def lp_nx(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = nx.read_gml(input, label='id')
    elif type(input) == IG:
        g = nx.from_scipy_sparse_matrix(igraph_to_sparse_matrix(input))
    else:
        g = input

    model = LabelPropagation(iterations=500)
    model.fit(g)
    community_dict = model.get_memberships()

    clusterID_relabel = {old_id: i for i, old_id in enumerate(set(community_dict.values()))}
    community_dict = {k: clusterID_relabel[v] for k, v in community_dict.items()}
    return commDict_to_commCluster(community_dict)


@add_labels
def gemsec(labels_true, input, cal_target='Q', weight=False):
    if type(input) == str:
        g = nx.read_gml(input, label='id')
    elif type(input) == IG:
        g = nx.from_scipy_sparse_matrix(igraph_to_sparse_matrix(input))
    else:
        g = input
    model = GEMSEC()
    model.fit(g)
    community_dict = model.get_memberships()
    return commDict_to_commCluster(community_dict)


@add_labels
def n2v_km_1(labels_true, input, cal_target='Q', weight=False):
    """
    node2vec+kmeans 最后计算模块度
    :param cal_target:
    :param weight:
    :return: Q
    """
    if type(input) == str:
        g = nx.read_gml(input, label='id')
    elif type(input) == IG:
        g = nx.from_scipy_sparse_matrix(igraph_to_sparse_matrix(input))
    else:
        g = input
    # emb_file=
    # node2vec
    embs = generate_node2vec_embeddings_1(g)
    num_cluster = len(set(labels_true))
    community = kmeans(X=embs, num_cluster=num_cluster)
    return community


# @add_labels
# def n2v_km(labels_true, input, cal_target='Q', weight=False):
#     """
#     node2vec+kmeans 最后计算模块度
#     :param cal_target:
#     :param weight:
#     :return: Q
#     """
#     if type(input) == str:
#         g = nx.read_gml(input, label='id')
#     elif type(input) == IG:
#         g = nx.from_scipy_sparse_matrix(igraph_to_sparse_matrix(input))
#     else:
#         g = input
#     # emb_file=
#     # node2vec
#     embs = generate_node2vec_embeddings(g)
#     num_cluster = len(set(labels_true))
#     community = kmeans(X=embs, num_cluster=num_cluster)
#     return community


community_method_dict = {'INF': infomap,
                         'LP': label_propagation,  # random
                         'FG': fastgreedy,
                         'WT': walktrap,
                         'LE': leading_eigenvector,
                         'LOU': multilevel,
                         'EB': edgebetweenness,
                         'n2v_km': n2v_km_1,
                         'scd': scd,
                         'lpnx': lp_nx,
                         'gemsec': gemsec}

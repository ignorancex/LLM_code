#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: linkpred_op.py
@time: 2020/7/14 16:49
@desc:
'''

import numpy as np
import networkx as nx
import copy
from igraph import Graph as IG
import scipy.sparse as sp

from utils.graph_op import igraph_to_sparse_matrix


# flag: refer to different type of graphs

# 局部指标
# CN
def CN(graph, flag=0):
    # 求共同邻居即求网络中2-hop路径的数量矩阵
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    # similarity_matrix = sp.triu(np.dot(adj, adj), k=1)
    similarity_matrix = sp.triu(adj.dot(adj), k=1, format='csr')
    return similarity_matrix


# Jaccard
def Jaccard(graph, flag=0):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    cn = sp.triu(adj.dot(adj), k=1)
    degree = np.asarray(adj.sum(axis=0))[0]
    x, y = cn.nonzero()
    scores = cn.data
    scores = scores / (degree[x] + degree[y] - scores)
    similarity_matrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape)
    return similarity_matrix.tocsr()


# Salton
def Salton(graph, flag=0):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    cn = sp.triu(adj.dot(adj), k=1)
    degree = np.asarray(adj.sum(axis=0))[0]
    x, y = cn.nonzero()
    scores = cn.data
    scores = scores / np.sqrt(degree[x] * degree[y])
    similarity_matrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape)
    return similarity_matrix.tocsr()


# Sorenson
# similar to Salton

# HPI
def HPI(graph, flag=0):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    cn = sp.triu(adj.dot(adj), k=1)
    degree = np.asarray(adj.sum(axis=0))[0]
    x, y = cn.nonzero()
    scores = cn.data
    dd = np.vstack((degree[x], degree[y]))
    minDegree = np.min(dd, axis=0)
    scores = scores / minDegree
    similarity_matrix = sp.coo_matrix((scores, (x, y)), shape=adj.shape)
    return similarity_matrix.tocsr()


# AA
def AA(graph, flag=0):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        A = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        A = igraph_to_sparse_matrix(graph).asformat("csr")

    A_ = A / np.log2(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = sp.triu(A.dot(A_), k=1, format='csr')
    return sim


def RA(graph, flag=0):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        A = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        A = igraph_to_sparse_matrix(graph).asformat("csr")

    A_ = A / A.sum(axis=1)
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = sp.triu(A.dot(A_), k=1, format='csr')
    return sim


# 局部指标
# LP
def LP(graph, flag=0, alpha=0.1):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    cn = adj @ adj
    triple_neighbors = alpha * adj @ adj @ adj
    lp = cn + triple_neighbors
    similarity_matrix = sp.triu(lp, k=1, format='csr')
    return similarity_matrix

# Katz
def Kate(graph, flag=0, beta=None):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    threshold = 1 / max(np.linalg.eig(adj.toarray())[0])
    # beta 必须小于邻接矩阵最大特征值的倒数
    beta = np.random.uniform(0, threshold)
    similarity_matrix = np.asarray(np.linalg.inv(np.eye(adj.shape[0]) - beta * adj) - np.eye(adj.shape[0]))
    return similarity_matrix


# ACT
def ACT(graph, flag=0, M=None):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        Laplac  = nx.laplacian_matrix(graph).toarray()
    elif flag == 1 and type(graph) == IG:
        Laplac  = np.array(IG.laplacian(graph))

    # pinv of L
    pinvL = np.linalg.pinv(Laplac)

    pinvL_diag = np.diag(pinvL)  # [deg(1), deg(2), ...]
    matrix_one = np.ones(shape=Laplac.shape)
    pinvL_xx = pinvL_diag * matrix_one
    similarity_matrix = pinvL_xx + pinvL_xx.T - (2 * pinvL)
    similarityMatrix = 1 / similarity_matrix
    similarityMatrix[similarityMatrix < 0] = 0
    return np.nan_to_num(similarityMatrix)


# RWR
def RWR(graph, flag=0, c=0.7):
    if flag == 0 and type(graph) == nx.classes.graph.Graph:
        adj = nx.adjacency_matrix(graph)
    elif flag == 1 and type(graph) == IG:
        adj = igraph_to_sparse_matrix(graph).asformat("csr")

    matrix_probs = adj / adj.sum(axis=0)
    matrix_probs = np.nan_to_num(matrix_probs)
    temp = np.eye(N=adj.shape[0]) - c * matrix_probs.T
    # matrix_rwr = (1 - c) * np.dot(np.linalg.inv(temp), np.eye(N=adj.shape[0]))
    try:
        matrix_rwr = (1 - c) * np.linalg.inv(temp)
    except:
        matrix_rwr = (1 - c) * np.linalg.pinv(temp)
    # except np.linalg.LinAlgError:
    similarityMatrix = matrix_rwr + matrix_rwr.T
    similarityMatrix[similarityMatrix < 0] = 0
    # return np.asarray(similarityMatrix)
    return sp.csr_matrix(similarityMatrix)

#
similarity_index_dict = {'cn': CN,
                         'jacc': Jaccard,
                         'salton': Salton,
                         'hpi': HPI,
                         'aa': AA,
                         'ra': RA,
                         'lp': LP,
                         'rwr': RWR}

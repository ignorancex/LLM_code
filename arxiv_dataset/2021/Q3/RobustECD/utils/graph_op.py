#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: graph_op.py
@time: 2020/7/14 16:25
@desc:
'''
import networkx as nx
import numpy as np
from igraph import Graph as IG
import scipy.sparse as sp
from utils.load_op import load_trueLabel


def graph_to_undirected_unweighted(g):
    if nx.is_directed(g):
        g = nx.to_undirected(g)
    atts = nx.get_node_attributes(g, 'commID')
    # if nx.is_weighted(g):
    #     weights = {edge: {'weight', 1} for edge in g.edges()}
    #     nx.set_edge_attributes(g, weights)
    adj = nx.adjacency_matrix(g)
    adj[adj > 1] = 1
    gg = nx.Graph(adj)
    nx.set_node_attributes(gg, atts, 'commID')
    return gg


def convert_networkx_to_igraph(g):
    """
    Convert networkx graph to python-igraph
    :param g: <class 'networkx.classes.graph.Graph'>
    :return: <class 'igraph.Graph'>
    """
    G = IG(directed=False)
    G.add_vertices(list(g.nodes()))
    G.add_edges(list(g.edges()))
    # add edge atts
    weights = nx.get_edge_attributes(g, 'weight')
    if weights != {}:
        G.es['weight'] = [1] * nx.number_of_edges(g)
        for edge in g.edges():
            G.es[edge]['weight'] = weights[edge]
    # add node atts
    atts = nx.get_node_attributes(g, 'commID')

    G.vs['commID'] = [0] * nx.number_of_nodes(g)
    for node in g.nodes():
        G.vs[node]['commID'] = atts[node]
    return G


def igraph_to_sparse_matrix(G):
    """
    get adj of igraph.Graph
    :param G: <class 'igraph.Graph'>
    :return:
    """
    shape = (G.vcount(), G.vcount())
    xs, ys = map(np.array, zip(*G.get_edgelist()))
    xs, ys = np.hstack((xs, ys)), np.hstack((ys, xs))
    return sp.coo_matrix((np.ones(xs.shape), (xs, ys)), shape=shape)


def get_nonedges_adj(g):
    '''
    get the matrix of nonedges
    :param g:
    :return:
    '''
    non_edges = list(nx.non_edges(g))
    xs, ys = map(np.array, zip(*non_edges))
    # xs, ys = np.hstack((xs, ys)), np.hstack((ys, xs))
    non_adj = sp.coo_matrix((np.ones(xs.shape), (xs, ys)), shape=(nx.number_of_nodes(g), nx.number_of_nodes(g)))
    return sp.triu(non_adj, k=1, format='csr')


def read_igraph(file):
    labels_true = load_trueLabel(file)
    atts = {node: {'value': label} for node, label in enumerate(labels_true)}
    g = nx.read_gml(file, label='id')
    g = graph_to_undirected_unweighted(g)
    nx.set_node_attributes(g, atts)

    # self.G = IG.Read_GML(self.file)
    return convert_networkx_to_igraph(g)


def rewire_graph(graph, add_edges, del_edges, save_path=False):
    """
    :param save_path:
    :param graph:  igraph
    :param add_edges: list([(source, target),...])
    :param del_edges: list([(source, target),...])
    :return:
    """
    # for edge in add_edges:
    graph.add_edges(add_edges)
    graph.delete_edges(del_edges)

    if save_path:
        graph.write_gml(save_path)
    return graph

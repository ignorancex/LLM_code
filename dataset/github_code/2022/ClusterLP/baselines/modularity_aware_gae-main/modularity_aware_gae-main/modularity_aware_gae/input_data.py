import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys


"""
Disclaimer: the parse_index_file function from this file, as well as the
cora/citeseer/pubmed parts of the loading functions, come from the
tkipf/gae original repository on Graph Autoencoders
"""


def parse_index_file(filename):

    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def load_data(dataset):

    """
    Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """
    if dataset == 'karate':
        dataset = 'karate_edges'
    file_name = 'data/'+dataset+'.txt'
    adj = nx.adjacency_matrix(nx.read_edgelist(file_name, delimiter = ' '))
    features = sp.identity(adj.shape[0])

    return adj, features


def load_labels(nodes_number):

    """
    Load node-level labels
    :param dataset: name of the input graph dataset
    :return: n-dim array of node labels, used for community detection
    """

    labels = np.repeat(range(1), nodes_number)

    return labels
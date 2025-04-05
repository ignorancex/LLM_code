#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: load_op.py
@time: 2020/7/14 15:14
@desc:
'''

import os
import networkx as nx
import numpy as np
import pandas as pd
from igraph import Graph as IG


def load_trueLabel(input):
    """
    load true label of dataset(nodes) from label file or graph
    :param input: bmname/ graph
    :return: list
    """
    global labels
    if type(input) == str:
        label_file = 'data/{}/{}_labels.txt'.format(input, input)
        context = np.loadtxt(label_file, dtype=np.int)
        labels = context[:, 1]
    elif type(input) == nx.classes.graph.Graph:
        labels = [""] * nx.number_of_nodes(input)
        for node in input.nodes():
            labels[node] = input.node[node]["value"]
    return labels


def load_lpm(lpm_list, mask):
    ind = np.array(list(map(lambda x: int(x), mask))).nonzero()[0]
    return np.array(lpm_list)[ind].tolist()


def clusterLabel_2_community(labels_pred):
    num_cluster = len(set(labels_pred))
    community = [''] * num_cluster
    for i in range(num_cluster):
        community[i] = []
    for node, commID in enumerate(labels_pred):
        # print(node, commID)
        community[commID].append(node)
    return community


def community_2_clusterDict(community):
    labels_pred_dict = {}
    for commID, comm in enumerate(community):
        for node in comm:
            labels_pred_dict[node] = commID
    return labels_pred_dict


class LargeGraphReader:
    def __init__(self, dataset):
        self.dataset = dataset
        root = 'data/large-net'
        self.dir = '{}/{}/'.format(root, dataset)
        self.files = os.listdir(self.dir)

        self.file_edges = self.dir + self.files[0]
        self.file_labels = self.dir + self.files[2]
        self.file_features = self.dir + self.files[1]

    def get_graph(self, feature=False):
        edges = pd.read_csv(self.file_edges)
        labels = pd.read_csv(self.file_labels)
        labels_dict = labels.set_index('id')['target'].to_dict()
        node_atts = {node: {'commID': commID} for node, commID in labels_dict.items()}

        graph = nx.convert_matrix.from_pandas_edgelist(edges, 'id_1', 'id_2')
        nx.set_node_attributes(graph, node_atts)

        sorted_graph = nx.convert_node_labels_to_integers(graph, ordering='default', label_attribute='old')

        return sorted_graph

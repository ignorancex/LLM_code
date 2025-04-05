#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: n2v_op.py
@time: 2020/7/20 18:25
@desc:
'''
import networkx as nx
import numpy as np
import copy
from gensim.models import Word2Vec

from utils.node2vec.src import node2vec

# def generate_node2vec_embeddings(graph, emd_size=128, emb_file=None):
#
#     node2vec = Node2Vec(graph, p=1, q=1, num_walks=10, workers=4)
#     model = node2vec.fit()
#     # save
#     if emb_file:
#         model.wv.save_word2vec_format(emb_file)
#
#     # get embs
#     embeddings = np.zeros([nx.number_of_nodes(graph), emd_size], dtype='float32')
#     for i in graph.nodes():
#         if str(i) in model.wv:
#             embeddings[i] = model.wv.word_vec(str(i))
#     return embeddings


def generate_node2vec_embeddings_1(graph, emd_size=128, emb_file=None):

    g = copy.deepcopy(graph)
    if not nx.is_weighted(g):
        attrs = {e: {'weight': 1} for e in g.edges()}
        nx.set_edge_attributes(g, attrs)

    G = node2vec.Graph(g, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, workers=4, iter=1)
    # save
    if emb_file:
        model.wv.save_word2vec_format(emb_file)

    # get embs
    embeddings = np.zeros([nx.number_of_nodes(g), emd_size], dtype='float32')
    for i in graph.nodes():
        if str(i) in model.wv:
            embeddings[i] = model.wv.word_vec(str(i))
    return embeddings
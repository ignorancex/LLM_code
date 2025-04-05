#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: REVSEL.py
@time: 2020/7/14 14:54
@desc: robust enhancement via vertex similarity ensemble learning
'''
import igraph as ig
from igraph import Graph as IG
import os
import math
from tqdm import tqdm
from functools import reduce
import warnings

warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger()
logger.setLevel("INFO")

from utils.load_op import *
from utils.graph_op import *
from utils.linkpred_op import *
from utils.communityDetection_op import *

similarity_list = ['cn', 'jacc', 'salton', 'hpi', 'aa', 'ra', 'lp', 'rwr']
CORE_COMM_SIZE_THRESHOLD = 3
LARGE_Nets = ['amazon-sub', 'dblp-sub']


class robustECD_VSEL():

    def __init__(self, bmname, cdm, sampleRatio, randomSample, threshold, sample_times_perIndex, useEdgeWeight=False, printINF=False, lpm=None,
                 lpm_mask=None):
        '''

        :param dataset:
        :param cdm: community detection method (S)
        :param sampleRatio:
        :param randomSample:
        :param threshold: pruning threshold
        :param lpm: vertex similarity
        :param lpm_mask: select lpm
        '''
        self.bmname = bmname

        # self.directed = directed
        self.cdm = cdm
        self.sampleRatio = sampleRatio
        self.randomSample = randomSample
        self.threshold = threshold
        self.lpm = lpm
        self.lpm_mask = lpm_mask
        self.sample_times_perIndex = sample_times_perIndex
        self.printINF = printINF

        # load
        if bmname not in LARGE_Nets:
            self.file = 'data/{}/{}.gml'.format(self.bmname, self.bmname)
        else:
            self.file = 'data/large-net/{}.gml'.format(self.bmname)
        # self.labels_true = load_trueLabel(self.bmname)
        # atts = {node: {'value': label} for node, label in enumerate(self.labels_true)}
        self.g = nx.read_gml(self.file, label='id')
        self.labels_true = list(nx.get_node_attributes(self.g, 'commID').values())
        self.g = graph_to_undirected_unweighted(self.g)

        # self.G = IG.Read_GML(self.file)
        self.G = convert_networkx_to_igraph(self.g)

        # save
        # self.output_path = output_path
        if printINF:
            logging.info('original graph basic inf:')
            logging.info('aver cluster coef: {}'.format(nx.average_clustering(self.g)))
            logging.info('=====================================\n')

    def get_similarity_rewire_scheme(self):
        '''
        compute vertex similarity matrix and get rewired edge schemes
        :return:  [edges_list1, edges_list2, ...]
        '''
        rewire_schemes = []
        self.simMatrix_list = []
        if self.printINF:
            logging.info('Cal vertex similarity...')

        if self.lpm:  # single lpm
            try:
                if self.printINF:
                    logging.info('cal {} ...'.format(self.lpm))
                s_matrix = similarity_index_dict[self.lpm](self.g)
                scheme = self.neg_sample(s_matrix)
                rewire_schemes += copy.deepcopy(scheme)
                self.simMatrix_list.append(s_matrix)
            except:
                if self.printINF:
                    logging.info('cal {} ... wrong!!!'.format(self.lpm))
                pass
        else:
            for lpm in load_lpm(lpm_list=similarity_list, mask=self.lpm_mask):
                try:
                    if self.printINF:
                        logging.info('cal {} ...'.format(lpm))
                    s_matrix = similarity_index_dict[lpm](self.g)
                    schemes = self.neg_sample(s_matrix)
                    rewire_schemes += copy.deepcopy(schemes)
                    self.simMatrix_list.append(s_matrix)
                except:
                    if self.printINF:
                        logging.info('cal {} ... wrong!!!'.format(lpm))
                    pass
        return rewire_schemes

    def neg_sample(self, similarity_matrix):
        '''
        sample according to similarity scores
        :param similarity_matrix:
        :return:
        '''
        nonedges_adj = get_nonedges_adj(self.g)
        # nonedges_sim = nonedges_adj * similarity_matrix
        nonedges_sim = nonedges_adj.multiply(similarity_matrix)
        xs, ys, data = sp.find(nonedges_sim)
        edge_and_score = list(zip(xs, ys, data))  # [(x, y, score),...]

        if not self.randomSample:
            # select top k
            edge_and_score_range = sorted(edge_and_score, key=lambda item: item[-1], reverse=True)
            edge_and_score_selected = edge_and_score_range[:int(math.ceil(nx.number_of_edges(self.g) * self.sampleRatio))]
            return [list(map(lambda x: x[:-1], edge_and_score_selected))]
        else:
            all_edges = []
            # weighted random sample
            allIndex = np.arange(len(edge_and_score))
            edge_and_score_arr = np.array(edge_and_score)
            scores = edge_and_score_arr[:, -1]
            scores = scores / sum(scores)
            for i in range(self.sample_times_perIndex):
                selectIndex = np.random.choice(a=allIndex, size=int(math.ceil(nx.number_of_edges(self.g) * self.sampleRatio)), replace=False,
                                               p=scores)
                edge_and_score_selected = edge_and_score_arr[selectIndex][:, :-1].astype(np.int)
                edge_and_score_selected = list(map(lambda x: tuple(x), edge_and_score_selected))
                all_edges.append(edge_and_score_selected)
            return all_edges

    def community_detection(self, rewire_schemes):
        self.channels = len(rewire_schemes)
        rewire_G = copy.deepcopy(self.G)

        communityCounterID = 1
        node_communityID_list = []  # entry = (node, commID)

        for i in range(self.channels):
            add_edges = rewire_schemes[i]
            # detect multiply edges
            try:
                assert set(rewire_G.get_edgelist()) & set(add_edges) == set()
            except:
                add_edges = list(set(add_edges) - (set(rewire_G.get_edgelist()) & set(add_edges)))

            # the id of new added edges
            newEdgeID = (rewire_G.ecount() + np.arange(len(add_edges))).tolist()
            # rewire graph and community detection
            rewire_G.add_edges(add_edges)

            communities = community_method_dict[self.cdm].__wrapped__(self.labels_true, input=rewire_G)
            node_communityID_list.extend([(node, j + communityCounterID) for j, comm in enumerate(communities) for node in comm])

            communityCounterID += len(communities)

            rewire_G.delete_edges(newEdgeID)
        # Constructing node-communities matrix, x: node, y:commIDs for per partition
        xs, ys = zip(*node_communityID_list)
        xs, ys = np.array(xs), np.array(ys)
        nodeCommunityMatrix = sp.coo_matrix((np.ones(xs.shape), (xs, ys)), dtype=np.int).tocsr()

        return nodeCommunityMatrix

    def aggregate_partition(self, nodeCommunityMatrix):
        # get co-occurrence adj
        co_matrix = nodeCommunityMatrix * nodeCommunityMatrix.T
        cooccurrence_matrix_dup = copy.deepcopy(co_matrix)
        # create co-occurrence graph
        co_matrix = sp.triu(co_matrix, k=1, format='csr')


        # threshold search
        if self.threshold != None:
            co_matrix.data[co_matrix.data < self.threshold] = 0
            co_matrix.eliminate_zeros()
            _, connect_components_labels = sp.csgraph.connected_components(csgraph=co_matrix, directed=False)
            cc = ig.Clustering(connect_components_labels)
        else:
            threshold, cc = threshold_search(co_matrix)
        # get core component
        core_components = [comm for comm in cc if len(comm) > CORE_COMM_SIZE_THRESHOLD]

        # get stray nodes
        core_nodes = reduce(lambda x, y: x + y, core_components)
        stray_nodes = [v for v in self.g.nodes() if v not in core_nodes]
        if self.printINF:
            logging.info('stray nodes: {}'.format(stray_nodes))

        # assign stray nodes to core components
        maxCommID = cal_averageSimilarity(core_components, self.simMatrix_list)

        # assign stray nodes to core components according to average similarity
        final_partition = copy.deepcopy(core_components)
        for strayNode in stray_nodes:
            final_partition[maxCommID[strayNode]].append(strayNode)
        return final_partition

    def run(self):

        rewire_schemes = self.get_similarity_rewire_scheme()
        nodeCommunityMatrix = self.community_detection(rewire_schemes)
        return self.aggregate_partition(nodeCommunityMatrix)

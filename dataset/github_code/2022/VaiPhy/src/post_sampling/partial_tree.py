import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import networkx as nx
from utils import rate_matrix, compute_loglikelihood, alphabet_size, nuc_vec


class PartialTree:
    def __init__(self, tree, n_leaves_forest, n_sites):
        self.tau = tree
        self.n_sites = n_sites
        self.nodes = sorted(self.tau.nodes())
        self.root = self.nodes[-1]
        self.leaves = []
        for node in self.nodes:
            if node < n_leaves_forest:
                self.leaves.append(node)

    def compute_up_messages(self, data):
        """ For a node u, the up_table[u] = P(observations under node u | node u). """
        # store up message for each node internal+external = 2n-2
        up_table = np.ones((len(self.tau), self.n_sites, alphabet_size))
        for i, leaf in enumerate(self.leaves):
            up_table[i] = np.array([nuc_vec[c] for c in data[leaf]])

        for node in nx.dfs_postorder_nodes(self.tau, self.root):
            if not node in self.leaves:
                node_idx = self.nodes.index(node)
                for neigh in self.tau._adj[node]:
                    if neigh < node:  # Neighbour is child of current node
                        neigh_idx = self.nodes.index(neigh)
                        t_child = self.tau.edges[(node, neigh)]['t']
                        trans_matrix = rate_matrix(t_child)
                        temp_table = np.dot(up_table[neigh_idx], trans_matrix)
                        up_table[node_idx] = np.multiply(up_table[node_idx], temp_table)
        return up_table

    def get_loglikelihood(self, data):
        up_table = self.compute_up_messages(data)
        return compute_loglikelihood(up_table)

######################################################################################
#
# Authors :
#
#           Email :
#
#####################################################################################
import os
import sys
sys.path.append(os.getcwd())

import logging
import numpy as np
import networkx as nx
from scipy.stats import beta
from utils import rate_matrix, alphabet_size, nuc_vec, optimize_t


class PhyloTree:
    def __init__(self, data):
        self.alphabet_size = alphabet_size  # number of letters in alphabet
        self.data = data
        self.n_leaves, self.n_sites = data.shape
        self.root = 2 * self.n_leaves - 3  # len(tree) - 1

    def initialize_leaf_links(self):
        up_table = np.zeros((self.n_leaves, self.n_sites, alphabet_size))
        n_leaves = self.n_leaves
        self.t_opts_leaves = np.zeros((n_leaves, n_leaves))
        self.W_leaves = np.full((n_leaves, n_leaves), -np.inf)
        for i in range(self.n_leaves):
            up_table[i] = np.array([nuc_vec[c] for c in self.data[i]])
        self.compute_t_opts_leaves(up_table, self.W_leaves, self.t_opts_leaves, self.n_leaves, self.n_sites)

    def compute_t_opts_leaves(self, up_table, W_leaves, t_opts_leaves, n_leaves, n_sites):
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                S_ij = np.zeros((alphabet_size, alphabet_size))

                for m in range(n_sites):
                    S_ij += np.outer(up_table[i, m], up_table[j, m])

                t_opt = optimize_t(S_ij)
                t_opts_leaves[i, j] = t_opt
                t_opts_leaves[j, i] = t_opt
                trans_matrix = rate_matrix(t_opt)
                temp = np.multiply(S_ij, np.log(trans_matrix))
                W_leaves[i, j] = np.sum(temp)
                W_leaves[j, i] = W_leaves[i, j]

    def compute_up_messages(self, tree):
        """ For a node u, the up_table[u] = P(observations under node u | node u). """
        # store up message for each node internal+external = 2n-2
        up_table = np.ones((len(tree), self.n_sites, self.alphabet_size))

        for i in range(self.n_leaves):
            up_table[i] = np.array([nuc_vec[c] for c in self.data[i]])

        for node in nx.dfs_postorder_nodes(tree, self.root):
            if not tree.nodes[node]['type'] == 'leaf':
                for child in tree.nodes[node]['children']:
                    t_child = tree.edges[(node, child)]['t']
                    trans_matrix = rate_matrix(t_child)
                    temp_table = np.dot(up_table[child], trans_matrix)
                    up_table[node] = np.multiply(up_table[node], temp_table)
        return up_table

    def compute_down_messages(self, tree, up_table):
        """ An alternative implementation. down_table[u] = p(Zu, Y_observations_above_u) """
        # Calculate mask values, a num_leaf x num_sites x alphabet matrix, where each observed value is 1.
        mask = np.ones((self.n_leaves, self.n_sites, alphabet_size))
        for i in range(self.n_leaves):
            mask[i] = np.array([nuc_vec[c] for c in self.data[i]])

        # store down message for each node internal+external = 2n-2
        down_table = np.ones((len(tree), self.n_sites, alphabet_size))
        down_table[self.root] /= self.alphabet_size

        for node in nx.dfs_preorder_nodes(tree, self.root):
            if not node == self.root:
                parent = tree.nodes[node]['parent']

                for child in tree.nodes[parent]['children']:
                    if child != node:
                        t_child = tree.edges[(parent, child)]['t']
                        trans_matrix = rate_matrix(t_child)
                        sibling_factor = np.dot(up_table[child], trans_matrix)
                        down_table[node] = np.multiply(down_table[node], sibling_factor)

                t_parent = tree.edges[(node, parent)]['t']
                trans_matrix = rate_matrix(t_parent)

                # If parent is an internal leaf, we need to restrict its value
                if parent < self.n_leaves:
                    parent_mask = np.multiply(mask[parent], down_table[parent])
                    down_table[node] = np.multiply(down_table[node], parent_mask)
                else:
                    down_table[node] = np.multiply(down_table[node], down_table[parent])

                down_table[node] = np.dot(down_table[node], trans_matrix.T)

        return down_table

    def compute_marginal_nodes(self, tree, up_table, down_table):
        logger = logging.getLogger('compute_marginal')

        n_nodes, n_sites, _ = up_table.shape
        nodes_marginal = np.ones((len(tree), self.n_sites, self.alphabet_size))
        for node in nx.dfs_preorder_nodes(tree, self.root):
            nodes_marginal[node] = np.multiply(up_table[node], down_table[node])

        log_lik_first_pos = np.log(np.sum(up_table[-1, 0]) / alphabet_size)

        for node in range(n_nodes):
            lik = np.sum(nodes_marginal[node, 0, :])
            log_lik = np.log(lik)
            # TODO We can remove this check / only include as a test, since it does unnecessary computations.
            diff = np.abs(np.exp(log_lik)-np.exp(log_lik_first_pos))
            if diff > 0.000001:
                logger.error("Marginal problem! Node: %s, log lik: %s, log lik from marg: %s, norm-scale diff: %s "
                             % (node, log_lik_first_pos, log_lik, diff))

        nodes_marginal /= nodes_marginal.sum(axis=-1, keepdims=True)
        return nodes_marginal

    def approx_count(self, node_i, node_j, nodes_marginal):
        # n_sites = len(nodes_marginal[0])
        # S_approx = np.zeros((4, 4))
        # for m in range(n_sites):
        #    S_approx += np.outer(nodes_marginal[node_i, m], nodes_marginal[node_j, m])
        return np.einsum("ij,ik->jk", nodes_marginal[node_i], nodes_marginal[node_j])

    def compute_w_ij(self, S_ij, trans_matrix):
        # Computes L_{local} as in SEM paper, i.e. an entry in W matrix.
        # S_ij and trans_matrix (parameterized by t) are a 4x4 matrices.
        # log_term = np.log(trans_matrix) - np.log(0.25)
        log_term = np.log(trans_matrix)
        l_local = np.sum(S_ij * log_term)
        return l_local

    def compute_edge_weights(self, n_nodes, g, t_opts, distort_S=False, poi_distortion_rate=0, return_S=False):
        """ The returned variable, weight_matrix is actually log_e_tilde:
            log_e_tilde = E_{g(Z)} [log P(Z, Y | T, theta)]
                        = sum_{u,v in A(T)} sum_{n,n' in Sigma} M^{u,v}_{n,n'} * log P_{lambda_{u,v}}(n, n') """
        # store message / expected count for each edge
        weight_matrix = np.full((n_nodes, n_nodes), -np.inf)
        n_leaves = self.n_leaves

        weight_matrix[:n_leaves, :n_leaves] = self.W_leaves

        # TODO repetitive calculation. We don't need to calculate leaf-to-leaf.
        S = np.zeros((n_nodes, n_nodes, 4, 4))
        for node1 in range(n_nodes):
            for node2 in range(node1 + 1, n_nodes):
                S_ij = self.approx_count(node1, node2, g)
                if distort_S:
                    conv = np.random.poisson(lam=poi_distortion_rate, size=4)
                    for i in range(4):
                        conv[i] = min(conv[i], S_ij[i, i])
                        S_ij[i, i] -= int(conv[i])
                        idx = (i + np.random.choice(3)) % 4
                        S_ij[i, idx] += int(conv[i])
                    S[node1, node2] = S_ij
                    S[node2, node1] = S_ij
                    if not np.allclose(np.sum(S_ij), self.n_sites):
                        print("Error! Mismatch in sites. ", self.n_sites, np.sum(S_ij), S_ij)

                trans_matrix = rate_matrix(t_opts[node1, node2])

                w_ij = self.compute_w_ij(S_ij, trans_matrix)
                weight_matrix[node1, node2] = w_ij
                weight_matrix[node2, node1] = w_ij
        if return_S:
            return weight_matrix, S
        return weight_matrix

    def compute_branch_lengths(self, n_nodes, g, cal_e, strategy, return_S=False):
        new_ts = np.zeros((n_nodes, n_nodes))
        n_leaves = self.n_leaves

        new_ts[:n_leaves, :n_leaves] = self.t_opts_leaves

        if return_S:
            S = np.zeros((*new_ts.shape, alphabet_size, alphabet_size))

        # TODO repetitive calculation. We don't need to calculate leaf-to-leaf.
        for node1 in range(n_nodes):
            for node2 in range(node1 + 1, n_nodes):
                S_ij = self.approx_count(node1, node2, g)
                if return_S:
                    S[node1, node2] = S_ij
                    S[node2, node1] = S_ij

                if strategy == 'ml':
                    t_opt = optimize_t(S_ij)  # TODO Add cal_e term?
                elif strategy == 'jc':
                    t_opt = sample_t_jc(S_ij)
                else:
                    logger = logging.getLogger('phylo.branch_length')
                    logger.error("Invalid branch length sampling strategy! %s" % strategy)
                    sys.exit()

                new_ts[node1, node2] = t_opt
                new_ts[node2, node1] = t_opt
        if return_S:
            return new_ts, S
        else:
            return new_ts

    def compute_g(self, g_prev, cal_e, t_opts):
        """ This function updates the g(Z) matrix, based on the derivation on Overleaf.
            It considers irregular trees, and works with p(Z, Y | T, theta) and uses cal_e matrix.
            Right now it is not an efficient implementation, we should use functions like einsum for performance. """

        logger = logging.getLogger('phylo.compute_g')

        g = g_prev.copy()
        K = g_prev.shape[2]  # alphabet size (4)
        n_nodes = 2 * self.n_leaves - 2

        order = range(self.n_leaves, 2 * self.n_leaves - 2)  # the one we used in some results

        for u in order:  # TODO change the order of updates

            g[u] = np.zeros_like(g[u])
            if u == self.root:
                log_root_term = np.log(np.ones_like(g[u]) / K)  # We assume uniform probability at root.
                g[u] += 1 * log_root_term  # g[u] in log-scale. cal_e of root, is 1 since it exists at every tree.

            for v in range(n_nodes):

                # If cal_e is zero, don't spend resources to calculate the rest.
                if cal_e[u, v] == 0 or v == u:
                    continue

                trans_matrix = rate_matrix(t_opts[u, v])  # alphabet x alphabet
                log_child_term = np.zeros_like(g[u])  # M x alphabet

                # If v is an observed node
                if v < self.n_leaves:
                    # print("v is observed")
                    for m in range(self.n_sites):
                        for i in range(K):
                            try:
                                j = np.where(g[v][m] == 1)[0][0]
                                temp_val = trans_matrix[j, i]
                            except:
                                temp_val = 1
                            log_child_term[m, i] = np.log(temp_val)
                    # print("\tchild term: ", np.exp(log_child_term))
                    # print("\tchildren term: ", np.exp(log_children_term))

                # If v is a latent node
                else:
                    # print("v is a latent node")
                    for m in range(self.n_sites):
                        for i in range(K):
                            temp_val = 1
                            for j in range(K):
                                temp_val *= np.power(trans_matrix[j, i], g[v][m, j])
                            log_child_term[m, i] = np.log(temp_val)

                    # print("\tchild term: ", np.exp(log_child_term))
                    # print("\tchildren term: ", np.exp(log_children_term))

                g[u] += cal_e[u, v] * log_child_term  # g[u] in log-scale

            max_g = np.max(g[u], keepdims=True)
            g[u] = np.exp(g[u] - max_g)
            g[u] /= np.sum(g[u], axis=-1, keepdims=True)
        return g


def f(p):
    return - 3 / 4 * np.log(4 / 3 * (p - 1 / 4))


def sample_t_jc(S_ij, return_q=False, return_p=False):
    m_same = np.trace(S_ij)
    m_diff = np.sum(S_ij) - m_same
    rv = beta(a=m_same+1, b=m_diff+1)
    p = rv.rvs(size=1)
    while p <= 0.25:
        p_ = rv.rvs(size=1000)
        if len(p_[p_ > 0.25]) == 0:
            continue
        else:
            p = p_[p_ > 0.25][0]
    t = f(p)
    if return_p:
        return t, p  #  t[0], p[0] Might need this in some cases because it tends to
    if return_q:
        # change of variables
        log_df_dt = - 4 / 3 * t
        log_q = rv.logpdf(p) + log_df_dt
        return t, log_q  # TODO might need to return first element here as well
    return t

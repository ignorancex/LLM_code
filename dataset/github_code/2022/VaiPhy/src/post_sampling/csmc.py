import os
import sys

sys.path.append(os.getcwd())

import time
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as sp_stats
from scipy.stats import beta
from scipy.special import logsumexp, comb, softmax
import scipy.special as sp_spec
from multiprocessing import Pool

from phylo_tree import PhyloTree, sample_t_jc
from data_utils import read_nex_file
from bifurcate_utils import bifurcate_tree
from bitmask_utils import retrieve_pair_partitions, retrieve_min_hamming_partition
from tree_utils import newick2nx, nx_to_newick, update_topology, save_consensus_tree
from resampling import stratified_resampling, multinomial_resampling, categorical_sampling
from post_sampling.partial_tree import PartialTree
import utils


# plt.switch_backend('agg')


class Particle:
    def __init__(self, idx, data, branch_sampling_scheme, tree_sampling_scheme, C1, log_pi_prior):
        self.T = nx.Graph()
        self.n_leaves, self.n_sites = data.shape
        self.n_nodes = 2 * self.n_leaves - 2
        self.roots = list(np.arange(self.n_leaves))
        self.T.add_nodes_from(self.roots)
        self.idx = idx
        self.log_w = 0  # un-normalized weights at each r
        self.branch_sampling_scheme = branch_sampling_scheme
        self.tree_sampling_scheme = tree_sampling_scheme
        self.log_branch_prior_r = 0
        self.log_forest_prior_r = 0
        self.log_pi = [log_pi_prior]  # stores pi_{s_rk} in csmc
        self.log_nu = 0  # v^+ in csmc
        self.log_nu_minus = 0  # v^- in csmc (overcounting correction term)
        self.C1 = C1  # normalizing constant for co-occurrences of internal-internal connections
        self.in_forest = []

    def sample_forest(self, bitmask_dict=None, merger_memory=None):
        internal_idx = len(self.T.nodes())
        if len(self.roots) == 3:
            add_roots = copy.deepcopy(self.roots)
            # Deterministic update w/ probability 1
            self.log_nu += 0
        else:
            if self.tree_sampling_scheme == "uniform":
                r1, r2 = np.random.choice(self.roots, size=2, replace=False)
                add_roots = [r1, r2]
                self.log_nu += np.log(1 / comb(len(self.roots), 2))  # Uniform, (num_roots choose 2)
            elif "bootstrap" in self.tree_sampling_scheme:  # == "naive_bootstrap":
                threshold = int(self.tree_sampling_scheme[-1])  # Assuming the names are like "naive_bootstrap_5"
                subtrees = [
                    PartialTree(tree=self.T.subgraph(c).copy(), n_leaves_forest=self.n_leaves, n_sites=self.n_sites)
                    for c in nx.connected_components(self.T)]
                merger_probs = []
                merger_combs = list(combinations(self.roots, 2))
                for root_pair in merger_combs:
                    node_i, node_j = root_pair
                    leaf_list = []
                    for i, partial_tree_obj in enumerate(subtrees):
                        if node_i in partial_tree_obj.nodes or node_j in partial_tree_obj.nodes:
                            leaf_list += partial_tree_obj.leaves

                    if "naive" in self.tree_sampling_scheme:
                        count = retrieve_pair_partitions(bitmask_dict, node_list=leaf_list,
                                                         max_difference=threshold)  # difference 0 if very strict, [1,n_taxa-2] range
                        merger_probs.append(np.log(count))
                    else:
                        _, log_count = retrieve_pair_partitions(bitmask_dict, node_list=leaf_list,
                                                                max_difference=threshold, return_LL=True)
                        merger_probs.append(log_count)
                # Normalize merger probabilities
                # print("merger_probs_before: ", merger_probs)
                if np.all(np.isinf(merger_probs)):  # if all elements are inf
                    merger_probs = np.zeros(len(merger_probs))
                # else:
                # print(merger_probs)
                merger_probs -= logsumexp(merger_probs)
                merger_probs = np.exp(merger_probs)

                merger_probs += 1e-10
                merger_probs /= np.sum(merger_probs)
                # print("merger_probs_after: ", merger_probs)
                # Sample the roots wrt their probabilities
                pair_idx = categorical_sampling(p=merger_probs)
                r1, r2 = merger_combs[pair_idx]
                add_roots = [r1, r2]
                self.log_nu += np.log(merger_probs[pair_idx])

            elif "naive_iw" in self.tree_sampling_scheme:
                subtrees = [
                    PartialTree(tree=self.T.subgraph(c).copy(), n_leaves_forest=self.n_leaves, n_sites=self.n_sites)
                    for c in nx.connected_components(self.T)]
                merger_probs = []
                merger_combs = list(combinations(self.roots, 2))
                for root_pair in merger_combs:
                    node_i, node_j = root_pair
                    leaf_list = []
                    for i, partial_tree_obj in enumerate(subtrees):
                        if node_i in partial_tree_obj.nodes or node_j in partial_tree_obj.nodes:
                            leaf_list += partial_tree_obj.leaves

                    mask_merger = np.zeros(self.n_leaves, dtype=int)
                    mask_merger[leaf_list] = 1
                    bitmask_merger = "".join(map(str, list(mask_merger)))

                    if bitmask_merger in merger_memory.keys():
                        iw = merger_memory[bitmask_merger]
                    else:
                        iw = []
                        for s in bitmask_dict.keys():
                            _, h_dist = retrieve_min_hamming_partition(bitmask_dict[s]['partitions'], bitmask_merger, leaf_list)
                            l_weight = bitmask_dict[s]['ll'] - bitmask_dict[s]['log_q'] - h_dist * np.log(10)
                            iw.append(l_weight)
                        iw = logsumexp(iw)
                        merger_memory[bitmask_merger] = iw
                    merger_probs.append(iw)
                if np.all(np.isinf(merger_probs)):  # if all elements are inf
                    merger_probs = np.zeros(len(merger_probs))

                # Normalize merger probabilities
                merger_probs -= logsumexp(merger_probs)
                merger_probs = np.exp(merger_probs)
                #print("merger_probs_after: ", sorted(merger_probs)[::-1])
                # Sample the roots wrt their probabilities
                pair_idx = categorical_sampling(p=merger_probs)
                r1, r2 = merger_combs[pair_idx]
                #print("n_roots: ", len(self.roots), "sampled ", r1, r2, " w prob: ", merger_probs[pair_idx], " unif: ", 1/len(merger_combs))
                add_roots = [r1, r2]
                self.log_nu += np.log(merger_probs[pair_idx])


            else:
                print("Invalid tree sampling scheme: ", self.tree_sampling_scheme)
                sys.exit()
        for r in add_roots:
            self.T.add_edge(internal_idx, r)
            self.roots.remove(r)
        self.roots.append(internal_idx)
        self.overcounting_correction()  # nu minus
        self.log_forest_prior_r = self.calculate_log_forest_prior()
        return add_roots, internal_idx

    def sample_forest_vcsmc(self):
        internal_idx = len(self.T.nodes())
        if len(self.roots) == 2:
            add_roots = copy.deepcopy(self.roots)
            # Deterministic update w/ probability 1
            self.log_nu += 0
        else:
            r1, r2 = np.random.choice(self.roots, size=2, replace=False)
            add_roots = [r1, r2]
            self.log_nu += np.log(1 / comb(len(self.roots), 2))  # Uniform, (num_roots choose 2)
        for r in add_roots:
            self.T.add_edge(internal_idx, r)
            self.roots.remove(r)
        self.roots.append(internal_idx)
        self.overcounting_correction()  # nu minus
        self.log_forest_prior_r = self.calculate_log_forest_prior()

    def sample_forest_weights(self, log_W):
        internal_idx = len(self.T.nodes())
        if len(self.roots) == 3:
            add_roots = copy.deepcopy(self.roots)
            # Deterministic update w/ probability 1
            self.log_nu += 0
        else:
            n_roots = len(self.roots)
            # merge roots r1 and r2
            # select r1 and r2 based on weights
            merge_probs = np.exp(log_W[self.roots])
            merge_probs = merge_probs / np.sum(merge_probs)  # normalize
            r1, r2 = np.random.choice(np.arange(0, merge_probs.shape[0]), size=2, p=merge_probs, replace=False)
            r1 = self.roots[np.mod(r1, n_roots)]
            r2 = self.roots[np.mod(r2, n_roots)]
            add_roots = [r1, r2]

        for r in add_roots:
            self.T.add_edge(internal_idx, r)
            self.roots.remove(r)
        self.roots.append(internal_idx)
        self.overcounting_correction()  # nu minus
        self.log_forest_prior_r = self.calculate_log_forest_prior()
        return (r1, r2), internal_idx

    def sample_forest_weights_mst(self, log_W):
        internal_idx = len(self.T.nodes())
        # merge roots r1 and r2
        # select r1 and r2 based on weights - too greedy, apply look-ahead?
        if len(self.roots) == 3:
            add_roots = copy.deepcopy(self.roots)
            # Deterministic update w/ probability 1
            self.log_nu += 0
        else:
            log_merge_probs = []
            log_merge_probs_2 = []
            n_roots = len(self.roots)
            n_nodes = log_W.shape[0]
            merges = []
            existing_edges = self.T.edges()
            for u, v in existing_edges:
                log_W[u, v] = 0
                log_W[v, u] = 0

            for i in range(n_roots):
                for j in range(i + 1, n_roots):
                    roots_ij = copy.deepcopy(self.roots)
                    r_i = self.roots[i]
                    r_j = self.roots[j]
                    merges.append((r_i, r_j))
                    roots_ij.remove(r_i)
                    roots_ij.remove(r_j)
                    roots_ij.append(internal_idx)
                    remaining_nodes = roots_ij + list(np.arange(internal_idx + 1, n_nodes))
                    log_W_ij_2 = log_W[np.ix_(remaining_nodes, remaining_nodes)]    # extracts submatrix
                    log_W_ij = log_W
                    log_W_ij[r_i, r_j] = 0
                    log_W_ij[r_j, r_i] = 0
                    G_ij_2 = nx.from_numpy_array(np.array(log_W_ij_2))
                    G_ij = nx.from_numpy_array(np.array(log_W_ij))
                    mst_ij_2 = nx.maximum_spanning_tree(G_ij_2)
                    mst_ij = nx.maximum_spanning_tree(G_ij)
                    log_prob_merge_ij_2 = mst_ij_2.size('weight')
                    log_prob_merge_ij = mst_ij.size('weight')

                    log_merge_probs_2.append(log_prob_merge_ij_2)
                    log_merge_probs.append(log_prob_merge_ij)

            normalized_log_merge_probs_2 = np.exp(log_merge_probs - logsumexp(log_merge_probs))
            normalized_log_merge_probs = np.exp(log_merge_probs_2 - logsumexp(log_merge_probs_2))

            idx = np.random.choice(np.arange(0, len(merges), dtype=int), p=normalized_log_merge_probs, replace=False)
            r1 = merges[idx][0]
            r2 = merges[idx][1]
            add_roots = [r1, r2]
            self.log_nu += normalized_log_merge_probs[idx]

        for r in add_roots:
            self.T.add_edge(internal_idx, r)
            self.roots.remove(r)
        self.roots.append(internal_idx)
        self.overcounting_correction()  # nu minus
        self.log_forest_prior_r = self.calculate_log_forest_prior()

        return add_roots, internal_idx

    def overcounting_correction(self):
        if self.log_nu != np.NINF:
            rho = len([r for r in self.roots if r >= self.n_leaves])
            self.log_nu_minus = np.log(1 / rho) if rho != 0 else 0
        else:
            self.log_nu_minus = np.NINF

    def sample_branch_lengths_vcsmc(self, branch_params, n_nodes, r):
        # Sample branch lengths
        root_idx = self.roots[-1]
        children = self.T.adj[root_idx].keys()
        log_branch_prob = 0
        for c, child in enumerate(children):
            q_dist = sp_stats.expon(scale=1 / branch_params[c][r])
            branch_sample = q_dist.rvs()
            self.T.adj[root_idx][child]['t'] = branch_sample

            log_branch_prob += q_dist.logpdf(branch_sample)  # p(b | tau, phi)

        self.log_nu += log_branch_prob
        self.log_branch_prior_r += log_branch_prob

    def sample_branch_lengths(self, phi, c):
        # sample branch lengths to the newly merged edges = [(root_idx, child) for child in children]
        root_idx = self.roots[-1]  # most recently added root
        children = self.T.adj[root_idx].keys()  # previous roots which merged with root_idx
        if self.branch_sampling_scheme == 'naive':
            # uniformly select jc distribution in the graph
            self.sample_branch_lengths_naive(self.n_nodes, children, root_idx, phi)
        elif self.branch_sampling_scheme == 'naive_w_labels':
            self.sample_branch_lengths_naive_w_labels(self.n_nodes, children, root_idx, phi, c)
        elif self.branch_sampling_scheme == 'same_parent':
            self.sample_branch_lengths_same_parent(self.n_nodes, children, root_idx, phi, c)
        elif self.branch_sampling_scheme == 'prior':
            self.sample_branch_prior(children, root_idx)

    def sample_branch_lengths_naive(self, n_nodes, children, root_idx, phi):
        prior = 2 / (n_nodes ** 2 - n_nodes)
        log_prob = np.log(prior) * np.log(len(children))
        for child in children:
            u, v = np.random.choice(np.arange(n_nodes), size=2, replace=False)
            phi_uv = phi[u, v]
            b, beta_draw = sample_t_jc(phi_uv, return_p=True)  # beta_draw comes from Beta, in [0,1]
            self.T.adj[root_idx][child]['t'] = b
            log_prob += self.jc_likelihood(b, beta_draw, phi)
            self.log_branch_prior_r += np.log(10) - 10 * b
        self.log_nu += log_prob

    def sample_branch_lengths_naive_w_labels(self, n_nodes, children, root_idx, phi, c, memorize=True):
        log_prob = 0
        for child in children:
            if child < self.n_leaves:
                u = child
                weight = np.zeros_like(c)
                w = c[u, self.n_leaves:] / np.sum(c[u, self.n_leaves:])
                # w = sample_gumbel_softmax(np.log(w), T=50)
                weight[u, self.n_leaves:] = w
                v = np.random.choice(np.arange(self.n_leaves, n_nodes), p=w)
            else:
                c_ = c.copy()
                if memorize:
                    for i, j in self.in_forest:
                        c_[i, j], c_[j, i] = 0, 0
                    self.C1 = np.sum(np.triu(c_[self.n_leaves:, self.n_leaves:], 1))
                c_internals = c_[self.n_leaves:, self.n_leaves:]
                weight = np.zeros_like(c_)
                weight[self.n_leaves:, self.n_leaves:] = c_internals / self.C1
                idx = np.random.choice(np.arange(n_nodes ** 2), p=np.ravel(np.triu(weight)))
                u, v = np.unravel_index(idx, (n_nodes, n_nodes))
            if memorize:
                self.in_forest.append((np.array([u]), np.array([v])))
            phi_uv = phi[u, v]
            b, beta_draw = sample_t_jc(phi_uv, return_p=True)  # beta_draw comes from Beta, in [0,1]
            self.T.adj[root_idx][child]['t'] = b
            log_prob_child = self.jc_likelihood(b, beta_draw, phi, weight=weight)
            log_prob += log_prob_child
            self.log_branch_prior_r += np.log(10) - 10 * b
        self.log_nu += log_prob
        # self.log_branch_prior_r += log_prob

    def sample_branch_lengths_same_parent(self, n_nodes, children, root_idx, phi, c, memorize=False):
        log_prob = 0
        # child 0
        child = list(children)[0]
        if child < self.n_leaves:
            u = child
            weight = np.zeros_like(c)
            w = c[u, self.n_leaves:] / np.sum(c[u, self.n_leaves:])
            weight[u, self.n_leaves:] = w
            v = np.random.choice(np.arange(self.n_leaves, n_nodes), p=w)
        else:
            c_ = c.copy()
            c_internals = c_[self.n_leaves:, self.n_leaves:]
            weight = np.zeros_like(c_)
            weight[self.n_leaves:, self.n_leaves:] = c_internals / self.C1
            idx = np.random.choice(np.arange(n_nodes ** 2), p=np.ravel(np.triu(weight)))
            u, v = np.unravel_index(idx, (n_nodes, n_nodes))
        phi_uv = phi[u, v]
        b, beta_draw = sample_t_jc(phi_uv, return_p=True)  # beta_draw comes from Beta, in [0,1]
        self.T.adj[root_idx][child]['t'] = b
        log_prob += self.jc_likelihood(b, beta_draw, phi, weight=weight)
        self.log_branch_prior_r += np.log(10) - 10 * b

        if u < self.n_leaves:
            parent = v
        else:
            parent = np.random.choice([u, v])

        for child in list(children)[1:]:
            if child < self.n_leaves:
                u = child
                v = parent
                weight = np.zeros_like(c)
                weight[u, v] = 1
            else:
                c_ = c.copy()
                w = c_[parent, self.n_leaves:] / np.sum(c_[parent, self.n_leaves:])
                weight = np.zeros_like(c_)
                weight[parent, self.n_leaves:] = w
                v = np.random.choice(np.arange(self.n_leaves, n_nodes), p=w)
                u = parent
                weight[u, self.n_leaves:] = w
                weight[self.n_leaves:, u] = w
                weight = np.triu(weight)
            phi_uv = phi[u, v]
            b, beta_draw = sample_t_jc(phi_uv, return_p=True)  # beta_draw comes from Beta, in [0,1]
            self.T.adj[root_idx][child]['t'] = b
            log_prob += self.jc_likelihood(b, beta_draw, phi, weight=weight)
            self.log_branch_prior_r += np.log(10) - 10 * b

        self.log_nu += log_prob
        # self.log_branch_prior_r += log_prob

    def sample_branch_prior(self, children, root_idx):
        log_prob = 0
        for child in children:
            b = np.random.exponential(1 / 10)
            self.T.adj[root_idx][child]['t'] = b
            log_prob_child = np.log(10) - 10 * b
            log_prob += log_prob_child
            self.log_branch_prior_r += np.log(10) - 10 * b
        self.log_nu += log_prob

    def sample_state(self, phi, r, c=None, bitmask_dict=None, merger_memory={}, method="vaiphy"):
        if method == "vaiphy":
            self.sample_forest(bitmask_dict=bitmask_dict, merger_memory=merger_memory)
            self.sample_branch_lengths(phi, c)
        elif method == "vcsmc":
            self.sample_forest_vcsmc()
            self.sample_branch_lengths_vcsmc(phi, self.n_nodes, r)

    def jc_likelihood(self, b, beta_draw, phi, weight=None):
        log_lik = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                w = 1 if weight is None else weight[i, j]
                if w == 0:
                    continue
                S_ij = phi[i, j, :, :]
                m_same = np.trace(S_ij)
                m_diff = np.sum(S_ij) - m_same
                # rv = beta(a=m_same + 1, b=m_diff + 1)
                a_param = m_same + 1
                b_param = m_diff + 1
                beta_draw_log_likelihood = utils.beta_log_pdf(beta_draw, a_param, b_param)
                # np.log(beta_draw**(a_param-1) * (1 - beta_draw)**(b_param-1) / sp_spec.beta(m_same+1, m_diff+1))
                log_df_dt = - 4 / 3 * b
                log_lik.append(np.log(w) + beta_draw_log_likelihood + log_df_dt)  # rv.logpdf(beta_draw) <-- costly
        return logsumexp(log_lik)

    def forest_likelihood(self, data):
        subtrees = [self.T.subgraph(c).copy() for c in nx.connected_components(self.T)]
        log_lik = []
        for i, partial_tree in enumerate(subtrees):
            partial_tree_obj = PartialTree(tree=partial_tree, n_leaves_forest=self.n_leaves, n_sites=self.n_sites)
            log_lik.append(partial_tree_obj.get_loglikelihood(data))
        return np.sum(log_lik)

    def calculate_log_forest_prior(self):
        log_forest_prior = 0
        subtrees = [self.T.subgraph(c).copy() for c in nx.connected_components(self.T)]
        for i, partial_tree in enumerate(subtrees):
            if len(partial_tree.nodes()) <= 1:
                n_leaves_i = 0
            else:
                n_leaves_i = len([x for x in partial_tree.nodes() if self.T.degree(x) == 1])
            if len(subtrees) == 1:
                log_forest_prior += -np.log(utils.double_factorial(2 * np.maximum(n_leaves_i, 3) - 5))
            else:
                log_forest_prior += -np.log(utils.double_factorial(2 * np.maximum(n_leaves_i, 2) - 3))
        return log_forest_prior

    def update_log_w(self, data):
        # Equation 4 in CSMC workshop paper: http://www.cs.columbia.edu/~amoretti/papers/phylo.pdf
        self.log_pi.append(self.forest_likelihood(data) + self.log_forest_prior_r + self.log_branch_prior_r)
        self.log_w = self.log_pi[-1] - self.log_pi[-2] + self.log_nu_minus - self.log_nu
        return self.log_w

    def add_branches(self, children, parent, branches_lengths_matrix, phi):
        log_prob = 0
        for c in children:
            b = branches_lengths_matrix[c, parent]
            self.T.adj[c][parent]['t'] = b
            m_same = np.trace(phi[c, parent])
            m_diff = np.sum(phi[c, parent]) - m_same
            log_prob += m_diff * np.log(1 / 4 - 1 / 4 * np.exp(-4 / 3 * b)) + m_same * np.log(1 / 4 + 3 / 4 * np.exp(-4 / 3 * b))

        self.log_branch_prior_r += log_prob

    def sample_branches_from_prior(self):
        log_prob = 0
        for e in self.T.edges():
            rate = 10
            b = 1 / 10  # np.random.exponential(1/rate)
            self.T.adj[e[0]][e[1]]['t'] = b
            self.log_branch_prior_r += -(rate * b) + np.log(rate)
            log_prob += -(rate * b) + np.log(rate)

        self.log_nu += log_prob
        return log_prob


def sample_gumbel_softmax(log_p, T):
    g = np.random.gumbel(0, 1, log_p.shape)
    y = softmax((g + log_p) / T)
    return y


def log_pi_prior_vcsmc(K):
    return np.log(1 / K)


def init_particles(K, data, branch_sampling_scheme, tree_sampling_scheme, C1):
    # returns np array of type object
    log_pi_prior = log_pi_prior_vcsmc(K)
    try:
        return np.array(
            [Particle(k, data, branch_sampling_scheme, tree_sampling_scheme, C1, log_pi_prior) for k in
             range(K)], dtype=object)
    except:
        return np.array(
            [Particle(k, data, branch_sampling_scheme, tree_sampling_scheme, C1, log_pi_prior) for k in
             range(K)], dtype=np.object)


def plot_path_trajectory(ancestor_path, filename=None, estimate=None):
    """
    Given ancestor paths, this function plots the trajectory of particles
    :param ancestor_path: list of lists. dimensions are approximately (R x K)
    :param filename: filename to save the image
    """
    R = len(ancestor_path)
    K = len(ancestor_path[0])
    plt.figure(figsize=(20, 5))
    # Place particles
    for r in range(R):
        plt.scatter(r * np.ones(K), np.arange(K), c='gray', alpha=0.5, s=1)
    # Plot trajectories
    for r in range(1, R):
        for dest, src in enumerate(ancestor_path[r]):
            plt.plot([r - 1, r], [src, dest], c='darkgray', alpha=0.3)
    # Plot surviving trajectories
    for k in range(K):
        i = k
        path_i = [i]
        for r in range(1, R)[::-1]:
            path_i.append(ancestor_path[r][i])
            i = ancestor_path[r][i]
        plt.plot(np.arange(R), path_i[::-1], c='royalblue', alpha=1)
    plt.xlabel("Rank (r)")
    plt.ylabel("Particles (k)")
    if estimate is None:
        plt.title("Trajectory of particles")
    else:
        plt.title("Trajectory of particles (Estimate %s)" % estimate)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def sample_trees(data, log_W, branch_lengths, K, phi, merger='uniform', beta=1):
    n_leaves = data.shape[0]
    R = n_leaves - 2  # rank, a function of taxa
    particles = init_particles(K, data, branch_sampling_scheme="None", tree_sampling_scheme="uniform", C1=None)
    ancestor_path = [np.arange(K)]
    log_Z_r = []
    log_q_R = np.zeros(K)
    log_pi = np.zeros((K, 2))
    log_pi[:, 0] = log_pi_prior_vcsmc(K)
    beta1 = beta             # tempering parameter
    beta2 = 1 - beta if beta != 1 else 1
    for r in range(R):
        log_w = np.zeros(K)
        if r >= 1 and 1 / np.exp(logsumexp(2 * log_w_tilde)) <= K:
            ancestor_idx = sorted(multinomial_resampling(np.exp(log_w_tilde)))
        else:
            # not sure how to handle importance weights when we don't resample? Hence thresh = K
            ancestor_idx = np.arange(K)
        ancestor_path.append(ancestor_idx)
        ancestor_particles = copy.deepcopy(particles)
        particles = []
        if r == R-1:
            # retain target at final rank
            beta1 = 1
            beta2 = 1

        for i, anc_idx in enumerate(ancestor_idx):
            particles.append(copy.deepcopy(ancestor_particles[anc_idx]))
            log_q_R[i] = log_q_R[anc_idx]
            log_pi[i] = log_pi[anc_idx]

        for k, particle in enumerate(particles):
            particle.log_nu = 0
            particle.log_nu_minus = 0
            if merger == 'mst':
                children, parent = particle.sample_forest_weights_mst(log_W)
            else:
                children, parent = particle.sample_forest()
            particle.add_branches(children, parent, branch_lengths, phi)
            log_pi[k, 1] = beta1 * particle.log_branch_prior_r + beta2 * particle.log_forest_prior_r
            log_w[k] += log_pi[k, 1] - log_pi[k, 0] - particle.log_nu + particle.log_nu_minus
            # log_w[k] = particle.update_log_w(data)
            log_q_R[k] += particle.log_nu - particle.log_nu_minus
            log_pi[k, 0] = log_pi[k, 1]
        log_w_tilde = log_w - logsumexp(log_w, keepdims=True)
        log_Z_r.append((logsumexp(log_w, keepdims=True)) - np.log(K))

    # sample from p(tau|phi, B)
    tau_idx = np.random.choice(np.arange(K, dtype=int), p=np.exp(log_w_tilde))
    log_Z = np.sum(np.array(log_Z_r))

    particle_lls = []
    tree_list = []
    for particle in particles:
        #particle_lls.append(particle.forest_likelihood(data))
        tree_list.append(particle.T)

    #plot_path_trajectory(ancestor_path)
    return tree_list[tau_idx], log_pi[tau_idx, 1], log_Z


def run(data, phi, K, c=None, bitmask_dict=None, method="vaiphy", branch_sampling_scheme='naive_w_labels',
        tree_sampling_scheme='uniform', seed_val=None, is_print=False, is_plot=False, output_directory=""):
    merger_memory = {}  # cache to be shared with other particles
    # c: unnormalized co-occurrences matrix
    start_time_total = time.time()
    if seed_val is not None:
        np.random.seed(seed_val)
    n_leaves = data.shape[0]
    # compute the normalizing constants for co-occurrences
    C1 = np.sum(np.triu(c[n_leaves:, n_leaves:], 1)) if c is not None else None
    R = n_leaves - 2  # rank, a function of taxa
    particles = init_particles(K, data, branch_sampling_scheme, tree_sampling_scheme, C1)
    ancestor_path = [np.arange(K)]
    log_Z_r = []
    for r in range(R):
        start_time = time.time()
        if is_print:
            print("\nr: ", r)

        # Resample
        if r >= 1 and 1 / np.exp(logsumexp(2 * log_w_tilde)) <= K:
            ancestor_idx = sorted(multinomial_resampling(np.exp(log_w_tilde)))
        else:
            # not sure how to handle importance weights when we don't resample? Hence thresh = K
            ancestor_idx = np.arange(K)
        ancestor_path.append(ancestor_idx)
        ancestor_particles = copy.deepcopy(particles)
        particles = []
        for i, anc_idx in enumerate(ancestor_idx):
            particles.append(copy.deepcopy(ancestor_particles[anc_idx]))

        log_w = np.zeros(K)
        for k, particle in enumerate(particles):
            particle.log_nu = 0
            particle.log_nu_minus = 0
            particle.sample_state(phi, r, c, bitmask_dict, merger_memory, method=method)
            log_w[k] = particle.update_log_w(data)
            if is_print:
                print("\tk: ", k, "\troots: ", particle.roots,
                      "\tlog_w: ", particle.log_w, "\tlog_nu: ",
                      particle.log_nu, "\tlog_pi: ", particle.log_pi, "\tlog_nu_minus", particle.log_nu_minus)
        log_w_tilde = log_w - logsumexp(log_w, keepdims=True)
        log_Z_r.append((logsumexp(log_w, keepdims=True)) - np.log(K))

        # log_Z_r.append((logsumexp(log_w[ancestor_idx], keepdims=True)) - np.log(K))

        if is_print:
            print("\tResampling: \tancestor_idx: ", ancestor_idx, "\tw_tilde: ", np.exp(log_w_tilde))
            for k, particle in enumerate(particles):
                print("\tk: ", k, "\troots: ", particle.roots)
        if True:
            print("\tTime taken by rank ", r, " (sec): ", time.time() - start_time, flush=True)
    log_Z = np.sum(np.array(log_Z_r))

    particle_lls = []
    for particle in particles:
        particle_lls.append(particle.forest_likelihood(data))

    if is_print:
        print("\nMarginal log likelihood estimate (Z_csmc): ", log_Z)
        print("\nParticle log likelihoods. Mean: ", np.mean(particle_lls), " Std: ", np.std(particle_lls),
              " Max: ", np.max(particle_lls))
        print("\nTotal time taken by CSMC (sec): ", time.time() - start_time_total)
    if is_plot:
        fname = output_directory + "_seed_" + str(seed_val) + ".png"
        plot_path_trajectory(ancestor_path, filename=fname, estimate=log_Z)
    return log_Z, particle_lls, particles


if __name__ == '__main__':
    print("Hello, world!")
    data_seed = 1  # 1, 2, 3
    num_taxa = 10  # 10, 20, 40
    n_sites = 300  # 300, 1000
    lambda_ = 4  # 0 if no distortion. 4 or 8 if distorted.
    root = 2 * num_taxa - 3

    K = 4
    csmc_n_repetitions = 1
    branch_sampling_scheme = 'naive_w_labels'  # 'naive', 'naive_w_labels', 'same_parent', 'prior'
    tree_sampling_scheme = 'naive_bootstrap_0'  # 'uniform', 'naive_bootstrap_X' where X =[0, 9], 'bootstrap_X', 'naive_iw'

    dataset_name = "data_seed_" + str(data_seed) + "_taxa_" + str(num_taxa) + "_pos_" + str(n_sites)
    results_dir = "../../results/" + dataset_name \
                  + "/model_vaiphy_init_nj_phyml_samp_slantis_branch_ml_ng_0.1" \
                  + "/S_8_seed_2"

    expected_count_fname = results_dir + "/post_analysis_prep" + "/exp_counts.npy"
    cooccurrence_fname = results_dir + "/post_analysis_prep" + "/node_cooccurrence_mat.npy"
    cooccurrence_fname_dist = results_dir + "/post_analysis_prep" \
                              + "/node_cooccurrence_mat_distorted_lambda_" + str(lambda_) + "_combined.npy"

    # Load Data
    fname = "../../data/" + dataset_name + "/" + dataset_name + ".nex"
    try:
        data = read_nex_file(fname)
        print("Data shape: ", data.shape)
    except:
        print("Problem occurred during loading data.", fname)
        sys.exit()

    try:
        phi = np.load(expected_count_fname)
        print("Phi shape: ", phi.shape)
    except:
        print("Problem occurred during loading phi.", expected_count_fname)
        sys.exit()

    if lambda_ == 0:
        cooccurrence_fname = results_dir + "/post_analysis_prep/node_cooccurrence_mat.npy"
    else:
        cooccurrence_fname = results_dir + "/post_analysis_prep/node_cooccurrence_mat_distorted_lambda_" \
                             + str(lambda_) + "_combined.npy"
    try:
        c = np.load(cooccurrence_fname)
    except:
        print("Problem occurred during loading co-occurrences.", cooccurrence_fname, lambda_)
        sys.exit()

    print("Distortion rate: ", lambda_)
    print("Co-occurrences shape: ", c.shape)

    bitmask_dict = None
    if lambda_ == 0:
        if tree_sampling_scheme == "naive_iw":
            bitmask_fname = results_dir + "/post_analysis_prep/bitmask_tree_dict.pkl"
        else:
            bitmask_fname = results_dir + "/post_analysis_prep/bitmask_dict.pkl"
    else:
        if tree_sampling_scheme == "naive_iw":
            bitmask_fname = results_dir + "/post_analysis_prep/bitmask_trees_dict_distorted_lambda_" \
                            + str(lambda_) + "_combined.pkl"
        else:
            bitmask_fname = results_dir + "/post_analysis_prep/bitmask_dict_distorted_lambda_" \
                            + str(lambda_) + "_combined.pkl"
    try:
        bitmask_dict = utils.load(bitmask_fname)
    except:
        bitmask_dict = None
        print("Problem occurred during loading bitmask dictionary.", bitmask_fname, lambda_)
        # sys.exit()
    print("Bitmask_dict: ", bitmask_dict)

    results_dir += "/csmc_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    print("Results will be saved to: %s" % results_dir)

    # Run csmc
    start_time_total = time.time()
    print("Running CSMC for data_seed: ", data_seed, " num_taxa", num_taxa, " n_sites: ", n_sites,
          " fname: ", expected_count_fname, " K: ", K)

    marg_logl_estimates = []

    out_fname = results_dir + "/branch_" + branch_sampling_scheme + "_tree_" + tree_sampling_scheme + "_K_" + str(K)
    if lambda_ != 0:
        out_fname += "_lambda_" + str(lambda_)

    """
    for rep in range(csmc_n_repetitions):
        start_time_rep = time.time()
        out_fname = results_dir + "/branch_" + branch_sampling_scheme + "_tree_" + tree_sampling_scheme + "_K_" + str(K)
        if lambda_ != 0:
            out_fname += "_lambda_" + str(lambda_)

        est, particle_lls = run(data=data, phi=phi, K=K, c=c, bitmask_dict=bitmask_dict,
                                branch_sampling_scheme=branch_sampling_scheme,
                                tree_sampling_scheme=tree_sampling_scheme, seed_val=rep,
                                is_plot=True, output_directory=out_fname)
        marg_logl_estimates.append(est)
        print("\tRepetition: \t", rep, "\t estimate: \t", est, "\t time taken: \t", time.time() - start_time_rep)
        print("\tLLs. Mean: ", np.mean(particle_lls), " Std: ", np.std(particle_lls), " Max: ", np.max(particle_lls))
    """
    with Pool(csmc_n_repetitions) as p:
        output = p.starmap(run,
                           [(data, phi, K, c, bitmask_dict, 'vaiphy', branch_sampling_scheme, tree_sampling_scheme, rep,
                             False, False, out_fname) for rep in range(csmc_n_repetitions)])
        all_particles = []
        all_particle_lls = []
        for est, particle_lls, particles in output:
            marg_logl_estimates.append(est)
            all_particle_lls += particle_lls
            all_particles += particles
            print("estimate: \t", est)
            # print("\tRepetition: \t", rep, "\t estimate: \t", est, "\t time taken: \t", time.time() - start_time_rep)
            print("\tLLs. Mean: ", np.mean(particle_lls), " Std: ", np.std(particle_lls), " Max: ",
                  np.max(particle_lls))

    print("\nMean marginal loglikelihood estimate (Z_csmc): ", np.mean(marg_logl_estimates))
    print("Std marginal loglikelihood estimate (Z_csmc): ", np.std(marg_logl_estimates))
    print("\nParticle LLs Mean: ", np.mean(all_particle_lls), " Std: ", np.std(all_particle_lls),
          " Max: ", np.max(all_particle_lls))
    print("Total time taken by CSMCS (sec): ", time.time() - start_time_total)

    best_particle_idx = np.argmax(all_particle_lls)
    best_tree = all_particles[best_particle_idx].T
    update_topology(best_tree, root)
    print("\nBest particle's LL: \t %s" % all_particle_lls[best_particle_idx])
    print("Best particle's Newick: \t %s" % nx_to_newick(best_tree, root))

    # Extract particle trees
    all_particle_trees = []
    all_particle_trees_bifurcated = []
    for particle in all_particles:
        cur_tree = particle.T
        update_topology(cur_tree, root)
        all_particle_trees.append(cur_tree)
        all_particle_trees_bifurcated.append(bifurcate_tree(cur_tree, num_taxa))

    # Save trees
    fname = out_fname + "_all_trees.pkl"
    utils.save(all_particle_trees, fname)
    print("\nParticle trees are saved to %s" % fname)

    phylo = PhyloTree(data)
    # Majority consensus
    for cutoff in [0.5, 0.1]:
        fname = out_fname + "_majority_consensus_cutoff_" + str(cutoff)
        consensus_newick = save_consensus_tree(fname, root=root,
                                               tree_list=all_particle_trees_bifurcated, cutoff=cutoff)
        print("\nConsensus tree Newick (cutoff=%s): \t%s" % (cutoff, consensus_newick))
        consensus_tree = newick2nx(consensus_newick, num_taxa)
        update_topology(consensus_tree, root)
        loglik = utils.compute_loglikelihood(phylo.compute_up_messages(consensus_tree))
        print("Consensus tree's Log-Likelihood (cutoff=%s): \t %s" % (cutoff, loglik))

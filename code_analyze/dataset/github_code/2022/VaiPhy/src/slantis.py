# Sequential Look-Ahead Non-Trans Importance Sampling (SLANTIS)
import os
import sys
sys.path.append(os.getcwd())

import logging
import numpy as np
import networkx as nx
from scipy.special import logsumexp
from resampling import categorical_sampling
from mst_utils import get_k_best_mst
from multiprocessing import Pool


class SLANTIS:
    def __init__(self, log_w, t_opts, explore_rate=0):
        self.explore_rate = explore_rate
        self.log_w = log_w
        self.t_opts = t_opts
        self.n = self.log_w.shape[0]
        self.n_leaves = (self.n + 2) // 2
        self.n_edges = self.n - 1
        self.n_internal_edges = self.n_edges - self.n_leaves
        self.S = nx.Graph()
        self.S_initial = None
        self.log_importance_weight = None
        self.R = self.log_w[self.n_leaves:][:, self.n_leaves:]
        self.M = []

    def sample_leaf_connections(self, log_w):
        self.log_importance_weight = 0
        for u in range(self.n_leaves):
            log_p = log_w[u, self.n_leaves:] - logsumexp(log_w[u, self.n_leaves:])
            p = np.exp(log_p)
            v = categorical_sampling(p) + self.n_leaves

            # add edge to S, account for probability of sampling it and remove it from W.
            self.S.add_edge(u, v, weight=self.log_w[u, v], t=self.t_opts[u, v])
            self.M.append((u, v))
            self.log_importance_weight += log_p[v - self.n_leaves]
            log_w[u, v], log_w[v, u] = -np.inf, -np.inf

    def remove_edges(self, fixed_edges, matrix):
        log_w = matrix.copy()
        log_w[fixed_edges[:, 0], fixed_edges[:, 1]] = -np.inf
        log_w.T[fixed_edges[:, 0], fixed_edges[:, 1]] = -np.inf
        return log_w

    def get_next_tree(self, alg="mst"):
        # R is an n_internals x n_internals graph
        mst = get_k_best_mst(self.R, k=1, t=np.zeros_like(self.R), alg=alg)[0]
        #if (not nx.is_connected(mst)) or len(mst.nodes()) != self.n - self.n_leaves:
        #    print("Warning! Cannot create a ST using R!")

        idx = np.array(mst.edges()).reshape((-1, 2))
        self.R = self.remove_edges(idx, self.R)
        tree_1 = nx.Graph()
        for edge in mst.edges():
            u, v = self.n_leaves + np.array(edge)
            tree_1.add_edge(u, v, weight=self.log_w[u, v], t=self.t_opts[u, v])
        return tree_1

    def check_cycles(self, S_alternative):
        lightest_edge = None
        try:
            cycle = nx.find_cycle(S_alternative)
            for edge_cycle in cycle:
                # If the edge is in M, skip it
                if edge_cycle in self.M:
                    pass
                # If the edge is not in M, proceed
                else:
                    if lightest_edge is None:
                        lightest_edge = edge_cycle
                        weight = S_alternative.get_edge_data(*lightest_edge)['weight']
                    else:
                        w = S_alternative.get_edge_data(*edge_cycle)['weight']
                        if w < weight:
                            lightest_edge = edge_cycle
                            weight = w

            if lightest_edge is not None:
                S_alternative.remove_edge(*lightest_edge)
        except nx.exception.NetworkXNoCycle:
            lightest_edge = None
        return S_alternative, lightest_edge

    def forward(self, tree_0, tree_1):
        tree_1_edges = list(tree_1.edges())
        sorted_idx = np.argsort([self.log_w[e] for e in tree_1_edges])[::-1]
        p_keeps = []
        for edge in tree_0.edges():

            # If edge is in S
            if self.S.has_edge(*edge):
                # Remove the edge from S, create a cut.
                S_alternative = self.S.copy()
                S_alternative.remove_edge(*edge)
                # Identify the components
                components = []
                for comp in nx.connected_components(S_alternative):
                    components.append(list(comp))
                # Find the heaviest edge in tree_1, that connects the cut.
                edge_alternative = None
                for sort_idx in sorted_idx:
                    e_temp = tree_1_edges[sort_idx]
                    # If the edge connects the two components, assign it to edge_alternative
                    if (e_temp[0] in components[0] and e_temp[1] in components[1]) \
                            or (e_temp[0] in components[1] and e_temp[1] in components[0]):
                        edge_alternative = e_temp
                        S_alternative.add_edge(*edge_alternative,
                                               weight=self.log_w[edge_alternative], t=self.t_opts[edge_alternative])
                        break

            # If edge is not in S
            else:
                # Add edge to S temporarily
                S_alternative = self.S.copy()
                S_alternative.add_edge(*edge, weight=self.log_w[edge], t=self.t_opts[edge])
                # If the edge creates a cycle, identify the worst edge in the cycle (which is not in M)
                S_alternative, lightest_edge = self.check_cycles(S_alternative)
                # If there is no edge to remove, skip
                if lightest_edge is None:
                    continue

            # Calculate probabilities of original and alternative S
            explore = np.random.uniform(0, 1)
            if explore > self.explore_rate:
                log_p_keep = self.S.size('weight')
                log_p_alternative = S_alternative.size('weight')
                log_normalizing_constant = np.logaddexp(log_p_keep, log_p_alternative)
                p_keep = np.exp(log_p_keep - log_normalizing_constant)
            else:
                p_keep = np.random.uniform(0, 1)
                log_p_keep = np.log(p_keep)
                log_normalizing_constant = 0

            p_keeps.append(p_keep)

            unif = np.random.uniform(0, 1)

            if self.S.has_edge(*edge):
                # If edge was in the original S, but there were no alternative edges to replace,
                # we chose original S with probability 1. p_keep = 1
                if edge_alternative is None:
                    self.M.append(edge)
                    #self.log_importance_weight += 0
                    #print("\tEdge ", edge, " is in S, we chose original S (no alternatives).\tp_keep: ", 1)
                # If edge was in the original S and we chose the alternative S
                elif p_keep < unif:
                    self.S = S_alternative
                    #print("\tEdge ", edge, " is in S, we chose alternative S.\tp_keep: ", p_keep)
                # If edge was in the original S and we chose the original S
                else:
                    self.M.append(edge)
                    self.log_importance_weight += log_p_keep - log_normalizing_constant
                    #print("\tEdge ", edge, " is in S, we chose original S.\tp_keep: ", p_keep)
            else:
                # If edge was in the alternative S and we chose the alternative S
                if p_keep < unif:
                    self.S = S_alternative
                    self.M.append(edge)
                    self.log_importance_weight += log_p_keep - log_normalizing_constant
                    #print("\tEdge ", edge, " is not in S, we chose alternative S.\tp_keep: ", p_keep)
                # If edge was in the alternative S and we chose the original S. Nothing changes.
                else:
                    #print("\tEdge ", edge, " is not in S, we chose original S.\tp_keep: ", p_keep)
                    pass

            if len(self.M) == self.n_edges:
                #print("The tree is sampled!")
                return

    def __call__(self, tree_0):
        logger = logging.getLogger('ots_call()')
        self.sample_leaf_connections(log_w=self.log_w.copy())
        for edge in tree_0.edges():
            self.S.add_edge(*edge, weight=self.log_w[edge], t=self.t_opts[edge])
        self.S_initial = self.S

        iter_ = 0
        propagate = True
        while propagate:
            #print("\nStarting round: ", iter_)
            #print("Length of M: ", len(self.M), " out of ", self.n_edges)

            # If tree_0 is not a ST, it is time to force the function to return a sample
            if len(tree_0.edges()) != self.n_internal_edges:  # TODO add networx function isconnected
                logger.warning("Warning! tree_0 is not a ST. Re-sampling the tree...") # TODO check the prob calculation for this case, since we didn't accept some scenarios
                self.S = None
                self.log_importance_weight = None
                break

            tree_1 = self.get_next_tree(alg="edmonds")
            self.forward(tree_0, tree_1)
            tree_0 = tree_1
            if len(self.M) == self.n_edges:
                break
            iter_ += 1
        return self.S


def sample_single_tree(log_w, t, idx, seed_base=0, explore_rate=0):
    np.random.seed(seed_base + idx)
    particles = []
    while len(particles) != 1:
        slantis = SLANTIS(np.copy(log_w), t, explore_rate=explore_rate)
        tree_start = slantis.get_next_tree("mst")
        slantis(tree_start)
        if slantis.S is not None:
            particles.append(slantis.S)
    return slantis.S, slantis.log_importance_weight


def sample_trees(log_w, t, n_particles=10, return_q=True, seed_base=0, explore_rate=0):
    # Sequential version, for debugging
    particles = [sample_single_tree(log_w, t, idx, seed_base, explore_rate) for idx, i in enumerate(np.arange(n_particles))]

    # Parallel version
    #with Pool(n_particles) as p:
    #    particles = p.starmap(sample_single_tree,
    #                          [(log_w, t, idx, seed_base) for idx, i in enumerate(np.arange(n_particles))])

    if return_q:
        return [particle[0] for particle in particles], np.array([particle[1] for particle in particles])
    return [particle[0] for particle in particles]

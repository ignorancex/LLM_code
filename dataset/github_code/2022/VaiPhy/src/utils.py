import os
import sys

sys.path.append(os.getcwd())

import pickle
import numpy as np
import scipy as sp
import networkx as nx
import scipy.special as sc
from numba import njit
import matplotlib.pyplot as plt

# define parameters
nuc_names = ['A', 'C', 'G', 'T']
transform = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': np.nan, '?': np.nan, ',': np.nan, 'N': np.nan, 'n': np.nan}
nuc_vec = {'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.], 'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
           '-': [1., 1., 1., 1.], '.': [1., 1., 1., 1.], '?': [1., 1., 1., 1.], 'N': [1., 1., 1., 1.],
           'n': [1., 1., 1., 1.]}
alphabet_size = len(nuc_names)
alpha = 1 / 3
stat_prob = np.array([1/alphabet_size] * alphabet_size)


# JC matrix decompose
def JC_param(alpha=.1):
    rate_matrix = alpha * np.ones((alphabet_size, alphabet_size))

    for i in range(alphabet_size):
        rate_matrix[i, i] = -3 * alpha

    D, U = np.linalg.eig(rate_matrix)
    U_inv = np.linalg.inv(U)

    return D, U, U_inv, rate_matrix


D, U, U_inv, Q = JC_param(alpha)


def rate_matrix(t):
    return np.dot(U * np.exp(D * t), U_inv)


def optimize_t(S_ij, max_branch_length=15):
    # One can use the d formula in https://www.ihes.fr/~carbone/MaximumLikelihood2.pdf to double-check (JC).
    # d = -0.75 log[ 1 - 4/3 * S_diff / (S_same + S_diff)]

    # optimizes branch length for maximum likelihood estimate. See notes for details
    # maximum separating branch length according to SEM code

    # First sum all counts where no state transitions occur, i.e. across the diagonal
    S_same = np.trace(S_ij)

    # Next sum counts where transitions have taken place
    S_diff = np.sum(S_ij) - S_same

    if S_diff == 0 and S_same == 0:
        raise ValueError('Both S_same and S_diff are zero')
    elif S_diff == 0:
        return (4 * alpha) * 1e-10
    elif 3 * S_same - S_diff <= 0:
        return max_branch_length

    u_bar = (3 * S_same - S_diff) / (3 * S_same + 3 * S_diff)

    t_opt = - np.log(u_bar) / (4 * alpha)
    return np.maximum(np.minimum(t_opt, max_branch_length), (4 * alpha) * 1e-10)


# simulate sequences given the tree topology and rate matrices
def simulate_seq(tree, ndata=10):
    n_nodes = len(tree)
    root = n_nodes - 1
    n_leaves = (n_nodes + 2) // 2
    pt_matrix = [np.zeros((alphabet_size, alphabet_size)) for i in range(2 * n_leaves - 3)]

    # do postorder tree traversal to compute the transition matrices
    for node in nx.dfs_postorder_nodes(tree, root):
        if not tree.nodes[node]['type'] == 'root':
            parent = tree.nodes[node]['parent']
            t = tree.edges[(parent, node)]['t']
            pt_matrix[node] = rate_matrix(t)

    simuData = []
    status = [''] * (2 * n_leaves - 2)
    for run in range(ndata):
        for node in nx.dfs_preorder_nodes(tree, root):
            if tree.nodes[node]['type'] == 'root':
                status[node] = np.random.choice(alphabet_size, size=1, p=stat_prob)[0]
            else:
                parent = tree.nodes[node]['parent']
                status[node] = np.random.choice(alphabet_size, size=1, p=pt_matrix[node][status[parent]])[0]

        simuData.append([nuc_names[i] for i in status[:n_leaves]])

    return np.transpose(simuData)


def compute_loglikelihood(up_table):
    """ Computes log-likelihood log P(observations | T, theta) = sum_m log P(observation at site m | T, theta).
     Note that it assumes uniform prior on the root. Otherwise, we should consider P(Xr).
    n_sites = len(up_table[0])
    log_likelihood = 0

    for pos in range(n_sites):
        log_likelihood += np.log(np.sum(up_table[-1, pos]) / alphabet_size)  # uniform root probability

    return log_likelihood
    """
    return np.sum(np.log(np.sum(up_table[-1] / alphabet_size, axis=-1)))


def save(content, file_name):
    """ Saves the content into pickle. """
    with open(file_name, 'wb') as fp:
        pickle.dump(content, fp)


def load(file_name):
    """ Loads the content from pickle. """
    with open(file_name, 'rb') as fp:
        content = pickle.load(fp)
    return content


def double_factorial(n):
    return sp.special.factorial2(n)


def get_experiment_setup_name(args):
    name = "model_" + args.model + "_init_" + args.init_strategy + "_samp_" + args.samp_strategy \
           + "_branch_" + args.branch_strategy + "_ng_" + str(args.ng_stepsize)
    return name


def beta_log_pdf(x, a, b):
    """
    Copied from scipy library
    """
    lPx = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x)
    lPx -= sc.betaln(a, b)
    return lPx


def check_tree_diversity(tree_list: list, logger=None):
    unique_edges = []
    edges_count = {}
    for i, tree in enumerate(tree_list):
        edges = tree.edges()
        if i == 0:
            unique_edges.append(list(edges))
            for e in edges:
                edges_count[e] = 1
        else:
            for e in edges:
                if (e[0], e[1]) in unique_edges:
                    edges_count[(e[0], e[1])] += 1
                elif (e[1], e[0]) in unique_edges:
                    edges_count[(e[1], e[0])] += 1
                else:
                    edges_count[e] = 1
                    unique_edges.append(e)

    print(f"N unique edges: {len(unique_edges)}")
    print(f"Unique edges: {unique_edges}")
    print(f"Edges count: {edges_count}")
    if logger is not None:
        logger.info(f"N unique edges: {len(unique_edges)}")
        logger.info(f"Unique edges: {unique_edges}")
        logger.info(f"Edges count: {edges_count}")
    #if not 'x_' in os.getcwd():
    #    counts_list = [value for (key, value) in edges_count]
    #    plt.hist(counts_list)
    #    plt.show()
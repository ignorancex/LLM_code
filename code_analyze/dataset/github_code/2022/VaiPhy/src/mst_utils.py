import os
import sys
sys.path.append(os.getcwd())

import logging
import numpy as np
import networkx as nx
from tree_utils import update_topology


def get_mst(W, t_opts):
    # W is a symmetric (2N - 1)x(2N - 1) matrix with MI entries. Last entry is link connection to root.
    G = nx.Graph()
    n_nodes = W.shape[0]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if W[i, j] == -np.infty:
                continue

            t = t_opts[i, j]  # nx.shortest_path_length(tree, i, j, weight='t')
            G.add_edge(i, j, weight=W[i, j], t=t)
    mst = nx.maximum_spanning_tree(G)
    return mst


def get_edmonds(W, t_opts):
    """ Given weight matrix and branch lengths, the algorithm first creates a directed graph
        (where root doesn't have incoming edge and observed nodes don't have outgoing edges).
        Then, Edmonds' algorithm is applied and a maximum weighted tree is returned. """
    # W is a symmetric (2N - 1)x(2N - 1) matrix. Last entry is link connection to root.
    n_nodes = W.shape[0]
    n_leaves = (n_nodes + 2) / 2
    root = n_nodes - 1

    # Create directed graph
    G = nx.DiGraph(directed=True)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # if W[i, j] == -np.infty:
            #     continue

            if i < n_leaves:
                if j >= n_leaves:
                    G.add_edge(j, i, weight=W[i, j], t=t_opts[i, j])

            elif i == root:
                G.add_edge(i, j, weight=W[i, j], t=t_opts[i, j])

            else:
                G.add_edge(j, i, weight=W[i, j], t=t_opts[i, j])
                if j != root:
                    G.add_edge(i, j, weight=W[i, j], t=t_opts[i, j])

    edm = nx.algorithms.tree.branchings.Edmonds(G)
    tree = edm.find_optimum(kind="max", style="arborescence", preserve_attrs=True)

    if False:
        # Check whether there are trans leaves or not.
        update_topology(tree, root)
        num_trans_leaf = 0
        num_trans_latent = 0
        for i in range(n_nodes):
            if i < n_leaves and tree._node[i]['type'] != 'leaf':
                num_trans_leaf += 1
            if i >= n_leaves and tree._node[i]['type'] == 'leaf':
                num_trans_latent += 1

        if num_trans_leaf != 0:
            logger = logging.getLogger('get_edmonds()')
            logger.error("Error! Edmonds tree has %s trans-leaf." % num_trans_leaf)

    return tree


def get_k_best_mst(W, k=3, t=None, mst=None, alg="mst"):
    """ This function returns k-best spanning trees, given k, weight and branch length matrices.
        If an initial tree is not provided, depending on the alg parameter, it starts with
        either the maximum spanning tree (mst) or maximum weighted tree from Edmonds Algorithm (edmonds).
        From these initial trees, the function finds k-1 best candidates and reports.

        Note: This is a strict (or limited) version of S-best algorithm.
        We make sure that none of the returned trees have observed nodes placed as internal nodes. """

    logger = logging.getLogger('k_best_mst_lim_Sbest()')

    list_mst = []
    if mst is None:
        if alg == "edmonds":
            mst = get_edmonds(W, t).to_undirected()  # Get spanning tree from Edmonds Algorithm
        else:  # alg == "mst":
            mst = get_mst(W, t)                      # Get maximum spanning tree
    list_mst.append(mst)

    if k == 1:
        return list_mst

    # compute weight matrix for the dual of the MST
    dual_W = W.copy()
    for edges in mst.edges:
        dual_W[edges] = -np.infty  # remove the current edge
        dual_W[edges[1], edges[0]] = - np.infty

    # compute the edge pairs to be swapped
    lambda_w = []
    e_f_pair = []
    for edge in mst.edges:
        mst_temp = mst.copy()
        mst_temp.remove_edge(*edge)
        components = tuple(nx.connected_components(mst_temp))
        for i in components[0]:
            for j in components[1]:
                if dual_W[i, j] == -np.infty:
                    continue
                lambda_w.append(W[edge] - dual_W[i, j])
                e_f_pair.append([edge, (i, j)])

    # Find the k-best edge pairs
    max_mst_id = np.argsort(np.array(lambda_w))
    max_mst_id = max_mst_id.astype(int)
    k_indx = max_mst_id[:]
    k_indx = k_indx.astype(int)

    n_nodes = W.shape[0]
    n_leaves = (n_nodes + 2) / 2
    root = n_nodes - 1

    non_selected_st = []

    # return the k-best pair
    for indx, i in enumerate(k_indx):
        mst_temp = mst.copy()
        mst_temp.remove_edge(*e_f_pair[i][0])
        mst_temp.add_edge(*e_f_pair[i][1],
                          weight=W[e_f_pair[i][1][0], e_f_pair[i][1][1]], t=t[e_f_pair[i][1][0], e_f_pair[i][1][1]])

        # Check the number of trans-leaf and trans-latents  # TODO in alg=MST, this part is problematic.
        update_topology(mst_temp, root)
        num_trans_leaf = 0
        num_trans_latent = 0
        for i in range(n_nodes):
            if i < n_leaves and mst_temp._node[i]['type'] != 'leaf':
                num_trans_leaf += 1
            if i >= n_leaves and mst_temp._node[i]['type'] == 'leaf':
                num_trans_latent += 1
        if num_trans_leaf == 0:
            list_mst.append(mst_temp)
        else:
            non_selected_st.append(mst_temp)

        if len(list_mst) == k:
            break

    if len(list_mst) != k:
        logger.warning("Error in limited STs! Couldn't find S spanning trees with no trans-leaves!")
        logger.warning("Adding remaining %s STs with trans-leaves..." % (k-len(list_mst)))

        for mst_temp in non_selected_st:
            list_mst.append(mst_temp)
            if len(list_mst) == k:
                break

    return list_mst

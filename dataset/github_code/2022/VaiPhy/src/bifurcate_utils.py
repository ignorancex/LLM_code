import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from tree_utils import update_topology


def deletion_step(tree, deleted, n_nodes):
    # deletion step (proposition 5.3 in SEM paper)
    n_leaves = (n_nodes + 2) // 2
    order = range(n_leaves, n_nodes)
    # order = np.random.permutation(list(range(n_leaves, n_nodes)))
    for j in order:
        if j in deleted:  # or j == root:
            continue
        d = tree.degree(j)
        if d == 1:
            # internal node is leaf
            tree.remove_node(j)
            deleted.append(j)
        elif d == 2:
            nbor_i, nbor_k = [(node, tree.adj[j][node]["t"]) for node in tree.adj[j]]
            t_new = nbor_i[1] + nbor_k[1]
            tree.add_edge(nbor_i[0], nbor_k[0], t=t_new)
            tree.remove_node(j)
            deleted.append(j)
    return deleted, tree


def insertion_step(tree, deleted, n_nodes, D, eps=1e-10):
    # insertion step (proposition 5.4)
    order = range(n_nodes)
    # order = np.random.permutation(list(range(n_nodes)))
    for i in order:
        if i in deleted:  # or i == root:
            continue
        d = tree.degree(i)
        if d > D[i]:
            try:
                j = deleted.pop()
            except IndexError:
                break
            nbors_i = [(node, tree.adj[i][node]["t"]) for node in tree.adj[i]]
            if D[i] == 3:
                idx = np.argsort([nbor[1] for nbor in nbors_i])
                tree.add_edge(i, j, t=eps)
                for id in idx[:2]:
                    tree.add_edge(nbors_i[id][0], j, t=nbors_i[id][1])
                    tree.remove_edge(nbors_i[id][0], i)
            else:
                # D[i] = 1
                tree.add_edge(i, j, t=eps)
                for nbor in nbors_i:
                    tree.add_edge(nbor[0], j, t=nbor[1])
                    tree.remove_edge(nbor[0], i)
    return deleted, tree


def bifurcate_tree(tree, n_taxa):
    leaves = list(range(n_taxa))
    root = 2 * n_taxa - 3

    new_tree = tree.copy()
    neighbors = new_tree.adj  # dict of neighbors and connecting weights
    n_nodes = len(neighbors)  # +1 for root
    D = [1 if n in leaves else 3 for n in range(n_nodes)]
    # D[root] = 2
    deleted = []
    not_bifurcated = True
    while not_bifurcated:
        deleted, new_tree = deletion_step(new_tree, deleted, n_nodes)
        if len(deleted) == 0 and np.all([new_tree.degree(n) == D[n] for n in new_tree]):
            break
        deleted, new_tree = insertion_step(new_tree, deleted, n_nodes, D)

    new_tree = update_topology(new_tree, root)
    return new_tree


def prune_tree(tree, n_taxa):
    """ Given a networkX tree with n_taxa, this function prunes the tree by
        i) removing latent nodes that are placed to leaves of the tree
        ii) removing latent nodes placed internal which have a single parent and a single child. """
    n_nodes = 2 * n_taxa - 2
    root = n_nodes - 1

    update_topology(tree, root)

    # delete latent nodes appearing as leaves
    is_changed = True
    while is_changed:
        is_changed = False
        for node_i in range(n_taxa, root):  # don't consider the root node
            try:
                if tree._node[node_i]['type'] == "leaf":
                    tree.remove_node(node_i)
                    is_changed = True
            except:
                continue
        update_topology(tree, root)

    # delete internal nodes with single child
    is_changed = True
    while is_changed:
        is_changed = False
        for node_i in range(n_taxa, root):  # don't consider the root node
            try:
                if len(tree._node[node_i]['children']) == 1:
                    cur_parent = tree._node[node_i]['parent']
                    cur_child = tree._node[node_i]['children'][0]
                    new_t = tree._node[node_i]['t'] + tree._node[cur_child]['t']  # combine branch lengths
                    tree.add_edge(cur_parent, cur_child, t=new_t)
                    tree.remove_node(node_i)
                    is_changed = True
                    update_topology(tree, root)
            except:
                continue

if False:
    from tree_utils import newick2nx, nx2ete, nx_to_newick, ete_compare
    true_newick = "(10:0.171183,((3:0.103089,14:0.026412)1:0.0052861,(0:0.079739,(2:0.275276,(17:0.011269,(11:0.0539145,(4:0.014454,12:0.0753199)1:0.103238)1:0.0459828)1:0.0324954)1:0.0637542)1:0.0976233)1:0.00947178,(((1:0.0192513,(8:0.0723893,9:0.173095)1:0.00974985)1:0.0789378,7:0.0018439)1:0.0495257,(((13:0.0185397,18:0.0346757)1:0.0432125,(6:0.0160832,(15:0.0681791,16:0.083417)1:0.191244)1:0.0535982)1:0.175014,(5:0.0500765,19:0.0852477)1:0.0937622)1:0.241526)1:0.0443541);"
    edmonds_newick = "(((10:0.174914,(((((12:0.0805515,4:0.0200685)1:0.0992892,11:0.0474965)1:0.0509837,17:0.00367357)1:0.0358997,2:0.255091)1:0.0568808,0:0.0652135)1:0.110079,((7:0.00755497,((9:0.166471,8:0.067969)1:0.0165765,1:0.0188706)1:0.0732303)1:0.0722052,((5:0.079717,19:0.105977)1:0.0438058,((13:0.0174548,18:0.037978)1:0.0492896,((16:0.0842349,15:0.0421339)1:0.203708,6:0.0118795)1:0.0482111)1:0.121008)1:0.227068)1:0.0501367)1:0.0100439)1:0.00771873,3:0.0846543,14:0.030372);"

    n_leaves = 20
    root = 2 * n_leaves - 3
    true_tree = newick2nx(true_newick, n_leaves)
    edmonds_tree = newick2nx(edmonds_newick, n_leaves)

    ete_res = ete_compare(nx2ete(true_tree, root), nx2ete(true_tree, root))
    print("True tree. RF: %s \tmax RF: %s \tnorm RF: %s" % (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))

    ete_res = ete_compare(nx2ete(edmonds_tree, root), nx2ete(true_tree, root))
    print("Edmonds tree. RF: %s \tmax RF: %s \tnorm RF: %s" % (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))
    print("common:\n", ete_res['common_edges'])
    print("edges in source:\n", ete_res['source_edges'])
    print("edges in ref:\n", ete_res['ref_edges'])

    edmonds_tree_bifurcated = bifurcate_tree(edmonds_tree, n_leaves)
    edmonds_tree_bifurcated_newick = nx_to_newick(edmonds_tree_bifurcated, root)
    print("New newick: ", edmonds_tree_bifurcated_newick)
    ete_res = ete_compare(nx2ete(edmonds_tree_bifurcated, root), nx2ete(true_tree, root))
    print("Edmonds tree bifur. RF: %s \tmax RF: %s \tnorm RF: %s" % (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))
    print("common:\n", ete_res['common_edges'])
    print("edges in source:\n", ete_res['source_edges'])
    print("edges in ref:\n", ete_res['ref_edges'])

    #prune_tree(edmonds_tree, n_leaves)

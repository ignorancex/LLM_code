import os
import sys
sys.path.append(os.getcwd())

import dendropy
import numpy as np
import networkx as nx
from Bio import Phylo
from Bio.Phylo.Consensus import majority_consensus
from ete3 import Tree
from io import StringIO
import matplotlib.pyplot as plt


def create_tree(n_leaves, scale=0.1, seed=None):
    # Seed for result reproducibility
    if seed is not None:
        np.random.seed(seed)

    n_nodes = 2 * n_leaves - 2
    tree = nx.Graph()
    node_names = [n for n in range(n_nodes)]
    leaves = node_names[:n_leaves]

    for node in leaves:
        tree.add_node(node, type='leaf')

    candidates = list(range(n_leaves))

    for node in node_names[n_leaves:]:
        if len(candidates) == 3:
            tree.add_node(node, type='root')

            child_1 = candidates[0]
            t_1 = np.random.exponential(scale)
            child_2 = candidates[1]
            t_2 = np.random.exponential(scale)
            child_3 = candidates[2]
            t_3 = np.random.exponential(scale)

            # update parent and child information
            tree.add_edge(node, child_1, t=t_1)
            tree.add_edge(node, child_2, t=t_2)
            tree.add_edge(node, child_3, t=t_3)
            tree.add_node(node, children=[child_1, child_2, child_3])
            tree.add_node(child_1, parent=node)  # add branch length to parent
            tree.add_node(child_2, parent=node)
            tree.add_node(child_3, parent=node)

        else:
            pair = np.random.choice(candidates, 2, replace=False)

            # connect children with node
            child_1 = pair[0]
            t_1 = np.random.exponential(scale)
            child_2 = pair[1]
            t_2 = np.random.exponential(scale)

            tree.add_node(node, type='internal')

            # update parent and child information
            tree.add_edge(node, child_1, t=t_1)
            tree.add_edge(node, child_2, t=t_2)
            tree.add_node(node, children=[child_1, child_2])
            tree.add_node(child_1, parent=node)  # add branch length to parent
            tree.add_node(child_2, parent=node)

            # remove nodes and add merged to the candidate
            candidates.remove(pair[0])
            candidates.remove(pair[1])
            candidates.append(node)
    return tree


def update_topology(tree, root, min_branch_length=1e-10):
    # add needed meta data to nodes: parent, children, type and branch length to parent
    n_nodes = len(tree)
    n_leaves = (n_nodes + 2) // 2
    preorder_nodes = list(nx.dfs_preorder_nodes(tree, root))
    visited = []

    for node in preorder_nodes:
        children = [child for child in tree.adj[node] if child not in visited]

        # root
        if node == root:
            tree._node[root].pop('parent', None)
            tree.add_node(node, type='root', children=children)
        else:
            # internal
            if len(children) != 0:
                tree.add_node(node, type='internal', children=children)
            # leaf
            else:
                tree.add_node(node, type='leaf')

        for child in children:
            t = tree.edges[(node, child)]['t']
            if t < min_branch_length:
                t = max(t, min_branch_length)
                tree.remove_edge(node, child)
                tree.add_edge(node, child, t=t)
            tree.add_node(child, t=t, parent=node)

        visited.append(node)

    return tree


def remove_edges(fixed_edges, matrix):
    log_w = matrix.copy()
    log_w[fixed_edges[:, 0], fixed_edges[:, 1]] = -np.inf
    log_w.T[fixed_edges[:, 0], fixed_edges[:, 1]] = -np.inf
    return log_w


def draw_dendrogram(tree, root):
    newick_str = nx_to_newick(tree, root)
    ete_tree = Tree(newick_str)
    print(ete_tree)


def save_tree_image(filename, newick_str, n_taxa=None):
    handle = StringIO(newick_str)
    tree = Phylo.read(handle, "newick")
    tree.ladderize()

    try:
        if n_taxa is not None:
            for clade in tree.get_terminals():
                if int(clade.name) >= n_taxa:
                    clade.color = 'red'
    except:
        pass

    fig = plt.figure(figsize=(20, 20))
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, show_confidence=False, do_show=False)
    fig.savefig(filename)
    plt.close()


def nx_to_newick(tree, root):
    newick_str = nx2ete(tree, root).write()
    # ETE adds extra root char, remove root, TBD: more neatly later
    newick_str = newick_str[1:len(newick_str)-5]+";"
    #print(newick_str)
    return newick_str


def nx2ete(graph, root):
    tree = Tree()
    # Setting up a root node for lvl-1 to attach to
    tree.add_child(name=root)
    # A copy in a list, because you may not want to edit the original graph
    edges = list(graph.edges)
    while len(edges) > 0:
        for parent, child in edges:
            t = graph.edges[(parent, child)]['t']
            # check if this edge's parent is in the tree
            added_tree_list = list(tree.traverse("preorder"))
            if len(added_tree_list) > 0:
                for leaf in added_tree_list:
                    if leaf.name == parent:
                        # if it is, add child and thus create an edge
                        leaf.add_child(name=child, dist=t)
                        # Wouldn't want to add the same edge twice, would you?
                        edges.remove((parent, child))
                    elif leaf.name == child:
                        # if it is, add child and thus create an edge
                        leaf.add_child(name=parent, dist=t)
                        # Wouldn't want to add the same edge twice, would you?
                        edges.remove((parent, child))
    # Now if there are edges still unplaced, try again.
    return tree


def newick2nx(newick_str, n_leaves, scale=0.1):
    """ Converts a newick string to Networkx graph. Newick -> Dendropy Tree -> NetworkX graph.
        TODO Beware! If the number of nodes in the newick string is greater than 2*taxa-2, it causes problem.
        It might create a cycle!!"""
    # Create Dendropy Tree object.
    dendro_tree = dendropy.Tree.get_from_string(newick_str, "newick")

    # Add taxa to internal nodes and convert leaf taxa to integers.
    root_visited = False
    n_nodes = 2 * n_leaves - 2
    node_idx = n_leaves
    for node in dendro_tree.preorder_node_iter():
        # Root
        if not root_visited:
            node.taxon = 2 * n_leaves - 3
            root_visited = True
        else:
            # Internal node
            if node.is_internal():
                node.taxon = node_idx
                node_idx = node_idx + 1
            # Leaf
            else:
                try:
                    node.taxon = int(str(node.taxon)[1:-1]) # Dendropy's leaf taxa has the form: 'name'. We take substring.
                except:
                    node.taxon = int(str(node.taxon)[2:-1])  # Sometimes the node names have "V" as well.

    # Convert Dendropy Tree to Networkx graph.
    tree = nx.Graph()
    node_names = [n for n in range(n_nodes)]
    leaves = node_names[:n_leaves]

    # Add nodes to the graph
    for node in leaves:
        tree.add_node(node, type='leaf')
    for node in node_names[n_leaves:-1]:
        tree.add_node(node, type='internal')
    tree.add_node(2 * n_leaves - 3, type='root')

    # Add edges to the graph
    for node in dendro_tree.preorder_node_iter():
        if node.is_internal():
            children = []
            for child_node in node.child_nodes():
                if child_node.edge_length is not None:
                    t = child_node.edge_length
                else:
                    t = np.random.exponential(scale)

                tree.add_edge(node.taxon, child_node.taxon, t=t)
                tree.add_node(child_node.taxon, parent=node.taxon)
                children.append(child_node.taxon)
            tree.add_node(node.taxon, children=children)
    return tree


def ete_compare(tree_1, tree_2):
    """ Compares two ete3 Trees.
    Returns rf, max_rf, norm_rf, effective_tree_size, ref_edges_in_source, source_edges_in_ref, source_subtrees,
    common_edges, source_edges, ref_edges and treeko_dist in dictionary format. """
    return tree_1.compare(tree_2)



def get_likelihood_from_newick_file(newick_filename, data):
    try:
        with open(newick_filename) as f:
            tree_newick = f.read()
            n_taxa = data.shape[0]
            nx_tree = newick2nx(tree_newick, n_taxa)
            update_topology(nx_tree, 2 * n_taxa - 3)

            from phylo_tree import PhyloTree
            from utils import compute_loglikelihood

            phylo = PhyloTree(data)
            up_table = phylo.compute_up_messages(nx_tree)
            loglikelihood = compute_loglikelihood(up_table)
    except:
        print("Problem occurred during loading true tree. ", newick_filename)
        loglikelihood = None
        nx_tree = None
    return loglikelihood, nx_tree


def save_consensus_tree(filename_prefix, root=None, tree_list=None, tree_newick_list=None, cutoff=0.5):
    if tree_list is not None:
        tree_newick_list = []
        for tree in tree_list:
            update_topology(tree, root)
            tree_newick_list.append(nx_to_newick(tree, root))

    tree_list_biopython = []
    for newick_str in tree_newick_list:
        handle = StringIO(newick_str)
        tree_list_biopython.append(Phylo.read(handle, "newick"))

    consensus_tree = majority_consensus(tree_list_biopython, cutoff=cutoff)

    fname = filename_prefix + ".nw"
    Phylo.write(consensus_tree, fname, 'newick')
    f = open(fname, "r")
    consensus_newick = f.readline()
    f.close()

    fname = filename_prefix + ".png"
    fig = plt.figure(figsize=(20, 20))
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(consensus_tree, axes=axes, show_confidence=True, do_show=False)
    fig.savefig(fname)
    plt.close()

    return consensus_newick


if False:  # TODO remove later
    tree_list = []
    tree_newick_list = []
    n_leaves = 10
    n_nodes = 18
    root = n_nodes - 1
    for i in range(20):
        tree = create_tree(n_leaves)
        update_topology(tree, root)
        tree_list.append(tree)
        newick_str = nx_to_newick(tree, root)
        tree_newick_list.append(newick_str)

    consensus_newick = save_consensus_tree(filename_prefix="aaa", root=root, tree_list=tree_list, tree_newick_list=None, cutoff=0)
    consensus_newick_2 = save_consensus_tree(filename_prefix="bbb", root=root, tree_list=None, tree_newick_list=tree_newick_list, cutoff=0.1)
    print(consensus_newick)

if False:
    consensus_newick = "(16:0.02955,18:0.02713,(20:0.02286,((((3:0.02835,(11:0.02987,(26:0.01416,(14:0.03056,(23:0.01963,(0:0.01491,(((17:0.01445,(10:0.01092,(19:0.01705,15:0.01172)100.00:0.00680)100.00:0.00805)100.00:0.03091,(6:0.02216,24:0.02765)100.00:0.01519)100.00:0.01253,(21:0.02014,9:0.02116)100.00:0.00792)100.00:0.00740)100.00:0.00000)100.00:0.02205)100.00:0.01312)100.00:0.01299)100.00:0.00824)100.00:0.00791,(25:0.01864,(1:0.01989,22:0.02246)100.00:0.00876)100.00:0.00621)100.00:0.00813,((8:0.02311,(2:0.03901,12:0.02644)100.00:0.00000)100.00:0.01134,(13:0.03116,4:0.03433)100.00:0.00000)100.00:0.00553)100.00:0.00000,(5:0.03773,7:0.03679)100.00:0.01350)100.00:0.00000)100.00:0.00000):0.00000;"
    print(consensus_newick)
    consensus_tree = newick2nx(consensus_newick, 27)

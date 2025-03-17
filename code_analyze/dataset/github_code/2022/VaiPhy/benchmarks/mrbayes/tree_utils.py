######################################################################################
#
# Authors :
#
#           Email :
#
# tree utils: Implements tree utility functions.
#####################################################################################


import dendropy
import numpy as np
import networkx as nx

from Bio import Phylo
from ete3 import Tree
from io import StringIO

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def create_example_tree():
    n_leaves = 6
    n_nodes = 2 * n_leaves - 2
    tree = nx.Graph()
    node_names = [n for n in range(n_nodes)]
    leaves = node_names[:n_leaves]
    for node in leaves:
        tree.add_node(node, type='leaf')

    for node in node_names[n_leaves:]:
        if node == n_nodes - 1:  # root
            tree.add_node(node, type='root')

            mat_b_a = np.array([[0.5, 0.2], [0.5, 0.8]])
            tree.add_edge(node, 6, t=mat_b_a)
            mat_c_a = np.array([[0.3, 0.2], [0.7, 0.8]])
            tree.add_edge(node, 7, t=mat_c_a)
            mat_j_a = np.array([[0.5, 0.3], [0.5, 0.7]])
            tree.add_edge(node, 5, t=mat_j_a)
            tree.add_node(node, children=[5, 6, 7])
            tree.add_node(5, parent=node)  # add branch length to parent
            tree.add_node(6, parent=node)
            tree.add_node(7, parent=node)

        else:  # internal
            if node == 6:  # b
                child_1 = 0
                mat_1 = np.array([[0.6, 0.7], [0.4, 0.3]])  # mat_d_b
                child_2 = 1
                mat_2 = np.array([[0.5, 0.8], [0.5, 0.2]])  # mat_e_b

            elif node == 7:  # c
                child_1 = 2
                mat_1 = np.array([[0.6, 0.6], [0.4, 0.4]])  # mat_f_c
                child_2 = 8
                mat_2 = np.array([[0.8, 0.2], [0.2, 0.8]])  # mat_g_c

            elif node == 8:  # g
                child_1 = 3
                mat_1 = np.array([[0.8, 0.1], [0.2, 0.9]])  # mat_h_g
                child_2 = 4
                mat_2 = np.array([[0.5, 0.5], [0.5, 0.5]])  # mat_i_g

            tree.add_node(node, type='internal')
            # update parent and child information
            tree.add_edge(node, child_1, t=mat_1)
            tree.add_edge(node, child_2, t=mat_2)
            tree.add_node(node, children=[child_1, child_2])
            tree.add_node(child_1, parent=node)  # add branch length to parent
            tree.add_node(child_2, parent=node)

    return tree


def create_tree(n_leaves, scale=0.1, seed=None):
    """
    :param n_leaves: Number of leaves
    :param scale: used for branch length
    :return: returns the tree
    """
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


def update_topology(tree, root, min_branch_length=0):
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


def draw_dendrogram(tree, root):
    newick_str = nx_to_newick(tree, root)
    ete_tree = Tree(newick_str)
    print(ete_tree)


def save_tree_image(filename, newick_str, n_taxa=None):
    handle = StringIO(newick_str)
    tree = Phylo.read(handle, "newick")
    tree.ladderize()

    if n_taxa is not None:
        for clade in tree.get_terminals():
            if int(clade.name) >= n_taxa:
                clade.color = 'red'

    fig = plt.figure(figsize=(20, 20))
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, show_confidence=False, do_show=False)
    fig.savefig(filename)
    plt.close()


def print_newick(tree, root):
    print(nx2ete(tree, root).write(format=3))


# TODO This method might not be working as expected.
def save_newick(tree, n_leaves, file_name="plots/tree.tre"):
    text_file = open(file_name, "w")
    text_file.write(nx2ete(tree, 2 * n_leaves - 3).write())
    text_file.close()


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


def robinson_foulds(tree_1, tree_2):
    """ Calculates Robinson-Foulds distance of two ete3 Trees. """
    res = tree_1.robinson_foulds(tree_2, unrooted_trees=True)  # TODO: Needs to be changed if we use the rooted trees.
    rf = res[0]
    rf_max = res[1]
    n_rf = rf / rf_max  # Normalised RF
    return rf, rf_max, n_rf


def ete_compare(tree_1, tree_2):
    """ Compares two ete3 Trees.
    Returns rf, max_rf, norm_rf, effective_tree_size, ref_edges_in_source, source_edges_in_ref, source_subtrees,
    common_edges, source_edges, ref_edges and treeko_dist in dictionary format. """
    return tree_1.compare(tree_2)


def dendropy_robinson_foulds(tree_1_newick, tree_2_newick, schema="newick"):
    """ Calculates Robinson-Foulds and weighted RF of two dendropy Trees, given newick strings. """
    tns = dendropy.TaxonNamespace()
    tree1 = dendropy.Tree.get(data=tree_1_newick, schema=schema, taxon_namespace=tns)
    tree2 = dendropy.Tree.get(data=tree_2_newick, schema=schema, taxon_namespace=tns)
    rf = dendropy.calculate.treecompare.symmetric_difference(tree1, tree2, is_bipartitions_updated=False)
    w_rf = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(tree1, tree2, is_bipartitions_updated=False)
    #print(tree1.as_ascii_plot())
    #print(tree2.as_ascii_plot())
    return rf, w_rf


def dendropy_false_positivies_and_negatives(ref_tree_1_newick, tree_2_newick, schema="newick"):
    """ Calculates the number of false positive and negative edges. ref_tree_1_newick is the true tree. """
    tns = dendropy.TaxonNamespace()
    tree1 = dendropy.Tree.get(data=ref_tree_1_newick, schema=schema, taxon_namespace=tns)
    tree2 = dendropy.Tree.get(data=tree_2_newick, schema=schema, taxon_namespace=tns)
    return dendropy.calculate.treecompare.false_positives_and_negatives(reference_tree=tree1, comparison_tree=tree2,
                                                                        is_bipartitions_updated=False)


def dendropy_euclidean_distance(tree_1_newick, tree_2_newick, schema="newick"):
    """ Calculates Euclidean distance (Felsenstein's 2004 'branch length distance') of two dendropy Trees,
    given newick strings. """
    tns = dendropy.TaxonNamespace()
    tree1 = dendropy.Tree.get(data=tree_1_newick, schema=schema, taxon_namespace=tns)
    tree2 = dendropy.Tree.get(data=tree_2_newick, schema=schema, taxon_namespace=tns)
    return dendropy.calculate.treecompare.euclidean_distance(tree1, tree2, is_bipartitions_updated=False)


def get_k_degree_graph(tree, k):
    """ Given a NetworkX tree and an integer k, this functions creates a connectivity matrix (n_nodes x n_nodes).
        If k=1, G is the adjacency matrix of the tree. """
    n_nodes = len(tree._node)
    distance_dict = dict(nx.shortest_path_length(tree))

    G = np.zeros((n_nodes, n_nodes), dtype=int)
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if distance_dict[u][v] <= k:
                G[u, v] = 1
                G[v, u] = 1

    print(np.sum(G), " out of ", n_nodes * (n_nodes - 1), " entries are full. ", 100 * np.sum(G) / (n_nodes * (n_nodes - 1)))
    return G

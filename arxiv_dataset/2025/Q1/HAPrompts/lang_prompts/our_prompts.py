"""
our proposed language prompts
"""
import nltk
from nltk.tree import Tree

# comparative prompts
# shared for generating
# (1) leaf-level peers comparative prompts
# (2) ancestor-level peers comparative prompts
def gen_comparative_lang_prompts(query_class, related_class):
    return f'How does {query_class} look differently from {related_class}?' 


# path-based generic prompts
def gen_path_lang_prompts(query_class, ancestor_class):
    prompts = []
    if ancestor_class is not None:
        prompts.append(f'What does {query_class} (a type of {ancestor_class}) look like?')
        prompts.append(f'Describe a picture of {query_class} (a type of {ancestor_class}).')
        prompts.append(f'What are the unique characteristics of {query_class} (a type of {ancestor_class})?')
    else: # a leaf class that is directly connected to the (artifical) root node
        prompts.append(f'What does {query_class} look like?')
        prompts.append(f'Describe a picture of {query_class}.')
        prompts.append(f'What are the unique characteristics of {query_class}?')
    return prompts



# utilities
def find_peer_nodes(tree, stats):
    if isinstance(tree, str): return
    for node in tree:
        complement = [n for n in tree if n != node]
        peers = [n.label() if isinstance(n, Tree) else n for n in complement]
        node_name = node.label() if isinstance(node, Tree) else node
        stats[node_name] = peers
        find_peer_nodes(node, stats)
    return


def root_to_leaf_path(tree, leaf):
    """
        return a list of tree nodes: [root, ..., leaf]
    """
    if type(tree) is nltk.Tree:
        path = []
        for node in tree:
            # recursion
            path = root_to_leaf_path(node, leaf)
            if len(path) != 0:
                break
        if len(path) != 0:
            path = [tree] + path
        else:
            path = []
    elif tree == leaf:  # base case 0
        path = [tree]
    else:  # base case 1
        path = []
    return path


def find_node_to_compare(peer_node, tree):
    """
        comparative prompts:
        (1) leaf-level peers
        (2) ancestor-level peers
    """
    leaves = tree.leaves()
    to_compare = {leaf:[] for leaf in leaves}
    for leaf in leaves:
        path = root_to_leaf_path(tree, leaf)
        path.reverse()
        path.pop() # remove root
        path_labels = [p.label() if isinstance(p,Tree) else p for p in path]
        for n in path_labels:
            to_compare[leaf] += peer_node[n]
    return to_compare


def find_node_to_compare_remove_leaf_comp(peer_node, tree):
    """
        comparative prompts:
        (2) ancestor-level peers
    """
    leaves = tree.leaves()
    to_compare = {leaf:[] for leaf in leaves}
    for leaf in leaves:
        path = root_to_leaf_path(tree, leaf)
        path.reverse() # after reverse, path = [leaf, ..., root]
        path.pop() # remove root
        path_labels = [p.label() if isinstance(p,Tree) else p for p in path]
        path_labels = path_labels[1:] # remove leaf
        for n in path_labels:
            to_compare[leaf] += peer_node[n]
    return to_compare


def find_node_to_compare_remove_nonleaf_comp(peer_node, tree):
    """
        comparative prompts:
        (1) leaf-level peers
        corner case: leaf class with no leaf-level peers 
                     -> substitute with ancestor-level peers
    """
    leaves = tree.leaves()
    to_compare = {leaf:[] for leaf in leaves}
    for leaf in leaves:
        to_compare[leaf] += peer_node[leaf]
    # fix corner case: leaf class has no leaf-level peers
    # for such cases, we resume to compare with non-leaf ancestor's peers
    for leaf in leaves:
        if len(to_compare[leaf]) == 0: # no leaf level peers
            path = root_to_leaf_path(tree, leaf)
            path.reverse() # after reverse, path = [leaf, ..., root]
            path.pop() # remove root
            path_labels = [p.label() if isinstance(p,Tree) else p for p in path]
            for n in path_labels:
                to_compare[leaf] += peer_node[n]
    return to_compare


def get_ancestors(tree):
    ancestors = {leaf: [] for leaf in tree.leaves()}
    for leaf in tree.leaves():
        # [root, ..., leaf]
        path = root_to_leaf_path(tree, leaf)
        path.pop() # remove leaf
        path.reverse()
        path.pop() # remove root
        path = [node.label() for node in path]
        if len(path) == 0: path.append(None)
        ancestors[leaf] = path
    return ancestors

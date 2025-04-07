# import statements
import REUParsing as rp
import numpy as np
import pickle
import conllu
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt

# for going through the files
import os
from tqdm import tqdm
import shutil

# keeps adjectives in
def ud_2_digraph_POS(tree, parent=1, graph=None): # changed name
    if graph is None:
        graph = nx.DiGraph()
        graph.add_node(graph.number_of_nodes(), name='root', upos='root') # what should the POS for the root be?
        graph.add_node(graph.number_of_nodes(), name=tree.token['form'], upos=tree.token['upos']) # added UPOS attribute to node (universal pos, not language specific)
        graph.add_edge(0, parent, deprel='<root>')
    for child in tree.children:
        child_num = graph.number_of_nodes()
        graph.add_node(child_num, name=child.token['form'], upos=child.token['upos']) # added upos
        graph.add_edge(parent, child_num, deprel=child.token['deprel'])
        graph = ud_2_digraph_POS(child, child_num, graph) # changed recursive call
    return graph


# gets rid of adjectives
# advmod discourse amod
def generate_BFShashes_no_modifiers(UDtrees, quiet=True): # changed name
    hashes = {}
    for sentence in (UDtrees if quiet else tqdm(UDtrees)):
        directed_tree = ud_2_digraph_POS(sentence)
        # edges_to_remove = []
        edges_to_add = []
        nodes_to_remove = []
        for edge in directed_tree.edges(data=True):
            # check if it is in a list of the modifiers to remove
            mod_list = ['advmod', 'discourse', 'amod']
            if edge[2]['deprel'] in mod_list:
                try:
                    s = directed_tree.successors(edge[1])
                    next_node = next(s)
                    dr = directed_tree.get_edge_data(edge[1], next_node)['deprel']
                    edges_to_add.append(tuple([edge[0], next_node, dr]))
                    nodes_to_remove.append(edge[1])
                    # edges_to_remove.append(tuple([edge[0], edge[1]]))
                    # edges_to_remove.append(tuple([edge[1], next_node]))
                except StopIteration:
                    nodes_to_remove.append(edge[1])
                    # edges_to_remove.append(tuple([edge[0], edge[1]]))
        # change the graph - add/remove edges/nodes
        for edge in edges_to_add:
            directed_tree.add_edge(edge[0], edge[1], deprel=edge[2])
        # for edge in edges_to_remove:
            # directed_tree.remove_edge(edge[0], edge[1])
        for node in nodes_to_remove:
            directed_tree.remove_node(node)
        # goes through nodes and creates the hashes
        for node in directed_tree.nodes(data=True):
            try:
                succ = directed_tree.successors(node[0])
                pred = directed_tree.predecessors(node[0])
                subgraph = pred_succ_subgraph(directed_tree, node[0], succ, pred)
                if not nx.is_empty(subgraph):
                    hash_ = nx.weisfeiler_lehman_graph_hash(subgraph,edge_attr='deprel',node_attr='upos') # added node_attr
                    if hash_ not in hashes:
                        hashes[hash_] = 0
                    hashes[hash_] += 1
            except AttributeError:
                pass
    return hashes


# generate hashes from digraph
def generate_hashes_POS(UDtrees, quiet=True): # changed name
    hashes = {}
    for sentence in (UDtrees if quiet else tqdm(UDtrees)):
        hash_ = nx.weisfeiler_lehman_graph_hash(ud_2_digraph_POS(sentence),edge_attr='deprel',node_attr='upos') # added node_attr
        if hash_ not in hashes:
            hashes[hash_] = []
        hashes[hash_].append(sentence)
    return hashes

# this is the code that DOES take into account the root or leaves
def generate_hashes_POS_2stepBFS(UDtrees, quiet=True): # changed name
    hashes = {}
    for sentence in (UDtrees if quiet else tqdm(UDtrees)):
        directed_tree = ud_2_digraph_POS(sentence)
        for node in directed_tree.nodes(data=True):
            try:
                succ = directed_tree.successors(node[0])
                pred = directed_tree.predecessors(node[0])
                subgraph = pred_succ_subgraph(directed_tree, node[0], succ, pred)
                if not nx.is_empty(subgraph):
                    hash_ = nx.weisfeiler_lehman_graph_hash(subgraph,edge_attr='deprel',node_attr='upos') # added node_attr
                    if hash_ not in hashes:
                        hashes[hash_] = 0
                    hashes[hash_] += 1
            except AttributeError:
                pass
    return hashes

# this is the code that DOES take into account the root or leaves
def pred_succ_subgraph(graph, node, succ, pred):
    ps_list = []
    ps_list.append(node)
    try:
        s = next(succ)
        ps_list.append(s)
        done_looping_s = False
        while not done_looping_s:
            try:
                ps_list.append(next(succ))
            except StopIteration:
                done_looping_s = True
    except StopIteration:
        pass
    try:
        p = next(pred)
        ps_list.append(p)
    except StopIteration:
        pass
    return graph.subgraph(ps_list)






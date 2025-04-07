import conllu
import networkx as nx
from conllu import parse_tree_incr
from tqdm import tqdm
from matplotlib import pyplot as plt

def load_sentences(file):
    with open(file, "r", encoding="utf-8") as data:
        sentences = parse_tree_incr(data)
        sentences = list(sentences)
    return sentences

def ud_2_graph(tree, parent=1, graph=None):
    if graph is None:
        graph = nx.Graph()
        graph.add_node(graph.number_of_nodes(), name='root', upos='ROOT')
        graph.add_node(graph.number_of_nodes(), name=tree.token['form'], upos=tree.token['upos'])
        graph.add_edge(0, parent, deprel='<root>')
    for child in tree.children:
        child_num = graph.number_of_nodes()
        graph.add_node(child_num, name=child.token['form'], upos=child.token['upos'])
        graph.add_edge(parent, child_num, deprel=child.token['deprel'])
        graph = ud_2_graph(child, child_num, graph)
    return graph

def draw_dep_tree(UDtree):
    g = ud_2_graph(UDtree)
    pos = nx.spring_layout(g)
    nx.draw(g, pos)
    nx.draw_networkx_labels(g, pos, labels={node:g.nodes[node]['name'] for node in g.nodes})
    nx.draw_networkx_edge_labels(g, pos, edge_labels={edge:g.edges[edge]['deprel'] for edge in g.edges})
    plt.show()

def get_hash(UDTree, edge_attr=None, node_attr=None):
    return nx.weisfeiler_lehman_graph_hash(ud_2_graph(UDTree),edge_attr=edge_attr, node_attr=node_attr)


def generate_hashes(UDtrees, edge_attr=None, node_attr=None, quiet=True):
    hashes = {}
    mostcommon = []
    for sentence in (UDtrees if quiet else tqdm(UDtrees)):
        hash_ = nx.weisfeiler_lehman_graph_hash(ud_2_graph(sentence),edge_attr=edge_attr, node_attr=node_attr)
        if hash_ not in hashes:
            hashes[hash_] = []
        hashes[hash_].append(sentence)
    return hashes

def diversity(UDtrees, edge_attr=None, node_attr=None):
    hashes = generate_hashes(UDtrees, edge_attr=edge_attr, node_attr=node_attr)
    if len(UDtrees) != 0:
        diversity = len(hashes)/len(UDtrees)
    else:
        diversity = -1
    return diversity

def compare_langs(langfile1, langfile2):
    pass
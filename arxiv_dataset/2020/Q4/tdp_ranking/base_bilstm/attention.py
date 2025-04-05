import numpy as np

from shared.attention import validate_node_words


def get_nodes(child_parent_candidates, max_words_per_node):
    nodes = set()
    for candidates_for_child in child_parent_candidates:
        for parent, child, label in candidates_for_child:
            nodes.add(parent)
            nodes.add(child)

    validate_node_words(nodes, max_words_per_node)
    sorted_nodes = sorted(nodes, key=lambda node: node.node_index_in_doc)
    return np.stack([[node.start_word_index_in_doc, node.end_word_index_in_doc] for node in sorted_nodes], axis=0)

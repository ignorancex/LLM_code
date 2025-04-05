import numpy as np

from shared.attention import validate_node_words


def get_nodes_array(sorted_nodes, max_words_per_node, word_in_doc_to_tokens_map):
    validate_node_words(sorted_nodes, max_words_per_node)
    return np.stack([node_to_array(node, word_in_doc_to_tokens_map) for node in sorted_nodes], axis=0)


def node_to_array(node, word_in_doc_to_tokens_map):
    return [
        # word_in_doc_to_tokens returns [start_index, end_index]
        word_in_doc_to_tokens_map[node.start_word_index_in_doc][0],
        word_in_doc_to_tokens_map[node.end_word_index_in_doc][1]
    ]

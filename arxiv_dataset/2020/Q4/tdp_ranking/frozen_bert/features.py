import numpy as np


def get_candidates_array(child_parent_candidates, word_in_doc_to_tokens_map):
    # Expect number of candidates to be the same for each child
    return np.array([[get_candidate_array(candidate, word_in_doc_to_tokens_map) for candidate in candidates_for_child]
                     for candidates_for_child in child_parent_candidates])


def get_candidate_array(candidate, word_in_doc_to_tokens_map):
    parent, child, _ = candidate

    # Should match FeatureLayer.candidate_array_length
    return np.array([
        parent.node_index_in_doc,
        # word_in_doc_to_tokens returns [start_index, end_index] for each word
        word_in_doc_to_tokens_map[parent.start_word_index_in_doc][0],
        word_in_doc_to_tokens_map[parent.end_word_index_in_doc][1],
        child.node_index_in_doc,
        word_in_doc_to_tokens_map[child.start_word_index_in_doc][0],
        word_in_doc_to_tokens_map[child.end_word_index_in_doc][1]
    ])

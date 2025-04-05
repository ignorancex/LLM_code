import numpy as np


def get_candidates_array(child_parent_candidates):
    # Expect number of candidates to be the same for each child
    return np.array([[get_candidate_array(candidate) for candidate in candidates_for_child]
                     for candidates_for_child in child_parent_candidates])


def get_candidate_array(candidate):
    parent, child, _ = candidate

    # Should match FeatureLayer.candidate_array_length
    return np.array([
        parent.node_index_in_doc,
        parent.start_word_index_in_doc,
        parent.end_word_index_in_doc,
        child.node_index_in_doc,
        child.start_word_index_in_doc,
        child.end_word_index_in_doc
    ])

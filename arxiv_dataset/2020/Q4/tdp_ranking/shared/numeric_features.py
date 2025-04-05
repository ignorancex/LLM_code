import numpy as np

from data_structures import is_padding_node

node_dist_feature_length = 5
sent_dist_feature_length = 2
label_feature_length = 3
padding_feature_length = 1
numeric_feature_length = node_dist_feature_length + sent_dist_feature_length + label_feature_length \
                         + padding_feature_length


def get_all_numeric_features(child_parent_candidates, node_count, sentence_count, TE_label_vocab):
    """Provide method to calculate this outside of model / gradient descent for efficiency"""

    return np.stack([get_numeric_features(parent, child, node_count, sentence_count, TE_label_vocab)
                     for candidates_for_child in child_parent_candidates
                     for parent, child, gold_label in candidates_for_child], axis=0)


def get_numeric_features(parent, child, node_count, sentence_count, TE_label_vocab):
    node_distance = get_node_distance_features(parent, child, node_count)
    same_sentence = get_sentence_distance_features(parent, child, sentence_count)
    label_combination = get_label_combination_features(parent, child, TE_label_vocab)
    padding = get_padding_features(parent)

    return np.concatenate([
        node_distance,
        same_sentence,
        label_combination,
        padding,
    ]).astype('float32')


def get_node_distance_features(p, c, node_count):
    """
    Return vector containing four one-hot entries representing three degrees of closeness if the parent is before
    the child, and one entry which is 1 if the parent is after the child. The last entry is a normalised distance.
    """

    node_distance = c.node_index_in_doc - p.node_index_in_doc
    same_sentence = c.snt_index_in_doc == p.snt_index_in_doc

    features = np.zeros((node_dist_feature_length,))
    if not is_padding_node(p):
        if node_distance == 1:
            # Adjacent
            features[0] = 1
        elif c.node_index_in_doc - p.node_index_in_doc > 1 and same_sentence:
            # Not adjacent but within same sentence is "closer" than different sentences
            features[1] = 1
        elif c.node_index_in_doc - p.node_index_in_doc > 1 and not same_sentence:
            features[2] = 1
        elif c.node_index_in_doc - p.node_index_in_doc < 1:
            features[3] = 1
        features[4] = node_distance / node_count  # Normalised node distance for document, between 0 and 1

    return features


def get_sentence_distance_features(p, c, sentence_count):
    sentence_distance = c.snt_index_in_doc - p.snt_index_in_doc

    features = np.zeros((sent_dist_feature_length,))
    if not is_padding_node(p):
        features[0] = 1 if sentence_distance == 0 else 0
        features[1] = sentence_distance / sentence_count  # Normalised distance for document, between 0 and 1

    return features


def get_label_combination_features(p, c, TE_label_vocab):
    """Return vector representing possible timex/event type labels, e.g. if both the parent and child are a time
    expression or are both an event"""

    timex_labels = [v for k, v in TE_label_vocab.items() if k.upper().startswith('TIMEX')]

    features = np.zeros((label_feature_length,))
    # If child is timex, correct parent must be root or timex
    if c.TE_label in timex_labels and p.TE_label == TE_label_vocab['ROOT']:
        features[0] = 1
    # If child is timex, correct parent must be root or timex
    if c.TE_label in timex_labels and p.TE_label in timex_labels:
        features[1] = 1
    # DCT is a very common parent for non-timexs
    if c.TE_label not in timex_labels and p.is_DCT:
        features[2] = 1
    return features


def get_padding_features(p):
    """Return feature indicating whether parent is a real node or just padding"""

    return np.array([1]) if is_padding_node(p) else np.array([0])

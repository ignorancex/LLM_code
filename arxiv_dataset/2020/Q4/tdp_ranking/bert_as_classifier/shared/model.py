import numpy as np
import tensorflow as tf

from data_structures import UNK_label
from shared.bert_layer import BertLayer
from shared.numeric_features import numeric_feature_length


def Model(max_sequence_length, size_edge_label, disable_handcrafted_features):
    # Inputs
    bert_input_ids = tf.keras.Input(shape=(max_sequence_length,), dtype=tf.int32, name='input_ids')
    bert_input_masks = tf.keras.Input(shape=(max_sequence_length,), dtype=tf.int32, name='input_masks')
    bert_segment_ids = tf.keras.Input(shape=(max_sequence_length,), dtype=tf.int32, name='segment_ids')
    numeric_features = tf.keras.Input(shape=(numeric_feature_length,), dtype=tf.float32, name='numeric_features')

    # Layers
    bert_layer = BertLayer('pooled_output', max_sequence_length, n_fine_tune_layers=1)
    concat_layer = tf.keras.layers.Concatenate()
    dense_layer = tf.keras.layers.Dense(size_edge_label)  # No activation, will softmax later

    # Assemble
    bert_outputs = bert_layer((bert_input_ids, bert_input_masks, bert_segment_ids))
    if disable_handcrafted_features:
        features = bert_outputs
    else:
        features = concat_layer([bert_outputs, numeric_features])
    scores = dense_layer(features)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids]
    else:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, numeric_features]
    return tf.keras.Model(inputs=inputs, outputs=[scores], name='feature_scoring_model')


def get_friendly_node_label(node, TE_label_set):
    """Use dictionary words instead of e.g. "timex" where possible to help with BERT embeddings"""

    if TE_label_set == 'none':
        return UNK_label
    elif TE_label_set == 'timex_event':
        if node.TE_label.upper().startswith('EVENT'):
            return 'event'
        elif node.TE_label.upper().startswith('TIMEX'):
            return 'time'
        else:
            # Is root, or padding
            return node.TE_label.lower()
    else:
        if node.full_label.upper().startswith('EVENT'):
            # If event, use event subclass without 'EVENT-' prefix
            return node.full_label.split('-')[1].lower()
        elif node.full_label.upper().startswith('TIMEX'):
            label_parts = node.full_label.split('-')
            if len(label_parts) > 1:
                # If timex, use timex subclass and prefix
                return 'time {}'.format(label_parts[1].lower())
            else:
                return 'time'
        else:
            # Is root, or padding
            return node.full_label.lower()


def get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set):
    """
    Of all the candidate parents for the child, only one of them has a label which is not 'NO_EDGE'.

    The model outputs a score for each candidate parent and edge label, indexed by parent and edge label.
    Assuming that is flattened down to list all the labels for parent 1, then all the labels for parent 2, etc.,
    return the index of the one parent and label combination which is correct i.e. not 'NO_EDGE'.
    """

    out_list = []
    for candidate in candidates_for_child:
        _, _, gold_label = candidate
        candidate_one_hot = []
        if labeled:
            for label in edge_label_set:
                if label == gold_label:
                    candidate_one_hot.append(1)
                else:
                    candidate_one_hot.append(0)
        elif not labeled:
            if gold_label != 'NO_EDGE':
                candidate_one_hot.append(1)
            else:
                candidate_one_hot.append(0)
        out_list.append(candidate_one_hot)

    return np.array(out_list)

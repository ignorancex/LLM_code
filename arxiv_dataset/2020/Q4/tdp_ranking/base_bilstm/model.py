import numpy as np
import tensorflow as tf

from base_bilstm.attention import get_nodes
from base_bilstm.embedding import EmbeddingLayer, get_words_and_labels_for_doc
from base_bilstm.features import get_candidates_array
from shared.attention import AttentionModel
from shared.features import candidate_array_length, numeric_feature_length
from shared.model import get_gold_candidate_one_hot
from shared.numeric_features import get_all_numeric_features
from shared.scoring import ScoringModel


def BiLSTMModel(size_word_vocab, size_TE_label_vocab, size_word_embed, size_TE_label_embed,
                TE_label_set, size_lstm):
    # CuDNNLSTM is much faster on GPU but is GPU-only
    LSTM = tf.keras.layers.CuDNNLSTM if tf.test.is_gpu_available() else tf.keras.layers.LSTM

    return tf.keras.Sequential([
        EmbeddingLayer(size_word_vocab, size_TE_label_vocab, size_word_embed, size_TE_label_embed,
                       TE_label_set),
        tf.keras.layers.Bidirectional(LSTM(size_lstm, return_sequences=True))
    ], name='bi_lstm_model')


def Model(size_word_vocab, size_TE_label_vocab, size_word_embed, size_lstm, size_feed_forward,
          TE_label_set, size_TE_label_embed, size_edge_label, max_words_per_node, max_candidate_count,
          disable_handcrafted_features):
    # Inputs
    word_label_ids = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='word_label_ids')
    nodes = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='nodes')
    candidates = tf.keras.Input(shape=(None, max_candidate_count, candidate_array_length), dtype=tf.int32,
                                name='candidates')
    numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32,
                                      name='numeric_features')

    # Sub-models
    bi_lstm_model = BiLSTMModel(size_word_vocab, size_TE_label_vocab, size_word_embed, size_TE_label_embed,
                                TE_label_set, size_lstm)
    size_bi_lstm = 2 * size_lstm

    attention_model = AttentionModel(size_bi_lstm, max_words_per_node)
    scoring_model = ScoringModel(size_bi_lstm, size_feed_forward, size_edge_label, max_candidate_count,
                                 disable_handcrafted_features)

    # Assemble pieces
    bi_lstm_output = bi_lstm_model(word_label_ids)
    attended_nodes = attention_model([nodes, bi_lstm_output])
    if disable_handcrafted_features:
        scoring_inputs = [bi_lstm_output, candidates, attended_nodes]
    else:
        scoring_inputs = [bi_lstm_output, candidates, numeric_features, attended_nodes]
    scores = scoring_model(scoring_inputs)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [word_label_ids, nodes, candidates]
    else:
        inputs = [word_label_ids, nodes, candidates, numeric_features]
    return tf.keras.Model(inputs=inputs, outputs=[scores])


def get_model_inputs(sentence_list, child_parent_candidates, word_vocab, TE_label_vocab, TE_label_set,
                     max_words_per_node, disable_handcrafted_features):
    word_label_ids = get_words_and_labels_for_doc(sentence_list, child_parent_candidates, word_vocab, TE_label_vocab,
                                                  TE_label_set)
    nodes = get_nodes(child_parent_candidates, max_words_per_node)
    candidates = get_candidates_array(child_parent_candidates)

    features_dict = {
        # Use extra wrapping list for batch dimension
        'word_label_ids': np.array([word_label_ids]),
        'nodes': np.array([nodes]),
        'candidates': np.array([candidates])
    }
    if not disable_handcrafted_features:
        numeric_features = get_all_numeric_features(child_parent_candidates, len(nodes),
                                                    len(sentence_list), TE_label_vocab)
        # Use extra wrapping list for batch dimension
        features_dict['numeric_features'] = np.array([numeric_features])
    return features_dict


def data_to_inputs_and_gold(data, labeled, word_vocab, TE_label_vocab, TE_label_set, edge_label_set,
                            max_words_per_node, disable_handcrafted_features):
    inputs_and_gold = []
    for document in data:
        sentence_list, child_parent_candidates = document
        model_inputs = get_model_inputs(sentence_list, child_parent_candidates, word_vocab,
                                        TE_label_vocab, TE_label_set, max_words_per_node,
                                        disable_handcrafted_features)
        # Use extra wrapping list for batch dimension
        gold_one_hots = np.array([[get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set)
                                   for candidates_for_child in child_parent_candidates]])
        inputs_and_gold.append((model_inputs, gold_one_hots))
    return inputs_and_gold


def split_inputs_and_gold(inputs_labels):
    inputs = [inputs for inputs, labels in inputs_labels]
    labels = [labels for inputs, labels in inputs_labels]

    return inputs, labels

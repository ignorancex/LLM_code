import tensorflow as tf

from frozen_glove.glove_embedding import GloVeEmbeddingLayer
from shared.attention import AttentionModel
from shared.features import candidate_array_length
from shared.numeric_features import numeric_feature_length
from shared.scoring import ScoringModel


def BiLSTMModel(size_TE_label_vocab, size_TE_label_embed,
                TE_label_set, size_lstm):
    # CuDNNLSTM is much faster on GPU but is GPU-only
    LSTM = tf.keras.layers.CuDNNLSTM if tf.test.is_gpu_available() else tf.keras.layers.LSTM

    return tf.keras.Sequential([
        GloVeEmbeddingLayer(size_TE_label_vocab, size_TE_label_embed, TE_label_set),
        tf.keras.layers.Bidirectional(LSTM(size_lstm, return_sequences=True))
    ], name='bi_lstm_model')


def Model(size_TE_label_vocab, size_lstm, size_feed_forward,
          TE_label_set, size_TE_label_embed, size_edge_label, max_words_per_node, max_candidate_count,
          disable_handcrafted_features):
    # Inputs
    word_label_ids = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='word_label_ids')
    nodes = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='nodes')
    candidates = tf.keras.Input(shape=(None, max_candidate_count, candidate_array_length), dtype=tf.int32,
                                name='candidates')
    if disable_handcrafted_features:
        numeric_features = None
    else:
        numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32,
                                          name='numeric_features')

    # Sub-models
    bi_lstm_model = BiLSTMModel(size_TE_label_vocab, size_TE_label_embed,
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

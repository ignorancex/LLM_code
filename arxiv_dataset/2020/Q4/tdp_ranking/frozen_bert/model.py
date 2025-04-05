import numpy as np
import tensorflow as tf

from frozen_bert.attention import get_nodes_array
from frozen_bert.features import get_candidates_array
from frozen_bert.te_embedding import get_TE_labels_for_doc, TEEmbeddingLayer
from shared.attention import AttentionModel
from shared.bert_layer import BatchedBertLayer, bert_output_size, InputExample, convert_examples_to_inputs
from shared.features import candidate_array_length, numeric_feature_length
from shared.model import get_gold_candidate_one_hot
from shared.numeric_features import get_all_numeric_features
from shared.scoring import ScoringModel


def Model(size_TE_label_vocab, size_TE_label_embed, TE_label_set, size_lstm, size_feed_forward, size_edge_label,
          max_sequence_length, max_words_per_node, max_candidate_count, disable_handcrafted_features):
    # Inputs
    TE_label_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='TE_label_ids')
    bert_input_ids = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='input_ids')
    bert_input_masks = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='input_masks')
    bert_segment_ids = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='segment_ids')
    nodes = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='nodes')
    candidates = tf.keras.Input(shape=(None, max_candidate_count, candidate_array_length), dtype=tf.int32,
                                name='candidates')
    numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32,
                                      name='numeric_features')

    # Sub-models
    TE_embedding_layer = TEEmbeddingLayer(size_TE_label_vocab, size_TE_label_embed, TE_label_set)
    # Split rest into inner model to ensure embeddings get calculated first (since non-differentiable)
    inner_model = InnerModel(size_TE_label_embed, size_lstm, size_feed_forward, size_edge_label,
                             max_sequence_length, max_words_per_node, max_candidate_count,
                             disable_handcrafted_features)

    # Assemble pieces
    TE_label_embeddings = TE_embedding_layer(TE_label_ids)
    if disable_handcrafted_features:
        inner_inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_embeddings, nodes, candidates]
    else:
        inner_inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_embeddings, nodes, candidates,
                        numeric_features]
    scores = inner_model(inner_inputs)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_ids, nodes, candidates]
    else:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_ids, nodes, candidates, numeric_features]
    return tf.keras.Model(inputs=inputs, outputs=[scores])


def InnerModel(size_TE_label_embed, size_lstm, size_feed_forward, size_edge_label,
               max_sequence_length, max_words_per_node, max_candidate_count, disable_handcrafted_features):
    # Inputs
    bert_input_ids = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='input_ids')
    bert_input_masks = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='input_masks')
    bert_segment_ids = tf.keras.Input(shape=(None, max_sequence_length), dtype=tf.int32, name='segment_ids')
    TE_label_embeddings = tf.keras.Input(shape=(None, size_TE_label_embed), dtype=tf.float32,
                                         name='TE_label_embeddings')
    nodes = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='nodes')
    candidates = tf.keras.Input(shape=(None, max_candidate_count, candidate_array_length), dtype=tf.int32,
                                name='candidates')
    numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32, name='numeric_features')

    # Sub-models/layers
    bert_layer = BatchedBertLayer('sequence_output', max_sequence_length, n_fine_tune_layers=0)
    sent_to_doc_reshape = tf.keras.layers.Reshape(target_shape=(-1, bert_output_size), name='bert_document_reshape')
    embedding_concat = tf.keras.layers.Concatenate(axis=-1)

    # CuDNNLSTM is much faster on GPU but is GPU-only
    LSTM = tf.keras.layers.CuDNNLSTM if tf.test.is_gpu_available() else tf.keras.layers.LSTM
    bi_lstm_layer = tf.keras.layers.Bidirectional(LSTM(size_lstm,
                                                       input_shape=(None, bert_output_size + size_TE_label_embed),
                                                       return_sequences=True))
    size_bi_lstm = 2 * size_lstm

    attention_model = AttentionModel(size_bi_lstm, max_words_per_node)
    scoring_model = ScoringModel(size_bi_lstm, size_feed_forward, size_edge_label, max_candidate_count,
                                 disable_handcrafted_features)

    # Assemble pieces
    bert_embeddings = bert_layer((bert_input_ids, bert_input_masks, bert_segment_ids))
    document_bert_embeddings = sent_to_doc_reshape(bert_embeddings)
    document_embeddings = embedding_concat([document_bert_embeddings, TE_label_embeddings])
    bi_lstm_output = bi_lstm_layer(document_embeddings)
    attended_nodes = attention_model([nodes, bi_lstm_output])

    if disable_handcrafted_features:
        scoring_inputs = [bi_lstm_output, candidates, attended_nodes]
    else:
        scoring_inputs = [bi_lstm_output, candidates, numeric_features, attended_nodes]
    scores = scoring_model(scoring_inputs)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_embeddings, nodes, candidates]
    else:
        inputs = [bert_input_ids, bert_input_masks, bert_segment_ids, TE_label_embeddings, nodes, candidates,
                  numeric_features]
    return tf.keras.Model(inputs=inputs, outputs=[scores], name='inner_model')


def get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates, TE_label_vocab, TE_label_set,
                     max_words_per_node, max_sequence_length, disable_handcrafted_features):
    sorted_nodes = get_sorted_nodes(child_parent_candidates)

    bert_inputs, word_in_doc_to_tokens_map = get_bert_inputs(bert_tokenizer, sentence_list, max_sequence_length)
    TE_label_ids = get_TE_labels_for_doc(len(sentence_list), sorted_nodes, word_in_doc_to_tokens_map, TE_label_vocab,
                                         TE_label_set, max_sequence_length)
    nodes = get_nodes_array(sorted_nodes, max_words_per_node, word_in_doc_to_tokens_map)
    candidates = get_candidates_array(child_parent_candidates, word_in_doc_to_tokens_map)

    features_dict = {
        # Use extra wrapping list for batch dimension
        'input_ids': np.array([bert_inputs['input_ids']]),
        'input_masks': np.array([bert_inputs['input_masks']]),
        'segment_ids': np.array([bert_inputs['segment_ids']]),
        'TE_label_ids': np.array([TE_label_ids]),
        'nodes': np.array([nodes]),
        'candidates': np.array([candidates])
    }

    if not disable_handcrafted_features:
        numeric_features = get_all_numeric_features(child_parent_candidates, len(nodes),
                                                    len(sentence_list), TE_label_vocab)
        # Use extra wrapping list for batch dimension
        features_dict['numeric_features'] = np.array([numeric_features])
    return features_dict


def get_bert_inputs(tokenizer, sentence_list, max_sequence_length):
    # Expect sentence to be a list of words
    examples = [InputExample(words_a=sentence) for sentence in sentence_list]

    inputs, words_to_tokens_maps = convert_examples_to_inputs(tokenizer, examples, max_sequence_length)

    word_in_doc_to_tokens_map = []
    all_input_ids = inputs['input_ids']
    token_length = 0

    for words_to_tokens_map, input_ids in zip(words_to_tokens_maps, all_input_ids):
        for word_to_tokens in words_to_tokens_map:
            start_token, end_token = word_to_tokens
            word_in_doc_to_tokens_map.append((token_length + start_token, token_length + end_token))
        token_length += len(input_ids)

    return inputs, word_in_doc_to_tokens_map


def get_sorted_nodes(child_parent_candidates):
    nodes = set()
    for candidates_for_child in child_parent_candidates:
        for parent, child, label in candidates_for_child:
            nodes.add(parent)
            nodes.add(child)

    return sorted(nodes, key=lambda node: node.node_index_in_doc)


def data_to_inputs_and_gold(data, bert_tokenizer, labeled, TE_label_vocab, TE_label_set, edge_label_set,
                            max_words_per_node, max_sequence_length, disable_handcrafted_features):
    inputs_and_gold = []
    for document in data:
        sentence_list, child_parent_candidates = document
        model_inputs = get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates,
                                        TE_label_vocab, TE_label_set, max_words_per_node, max_sequence_length,
                                        disable_handcrafted_features)
        # Use extra wrapping list for batch dimension
        gold_one_hots = np.array([[get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set)
                                   for candidates_for_child in child_parent_candidates]])
        inputs_and_gold.append((model_inputs, gold_one_hots))
    return inputs_and_gold

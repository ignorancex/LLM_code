import tensorflow as tf

from shared.features import candidate_array_length, numeric_feature_length, FeatureModel, get_feature_length


def FeatureScoringModel(size_feed_forward, size_edge_label, feature_length):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(size_feed_forward, use_bias=True, bias_initializer='glorot_uniform',
                              activation=tf.nn.relu, input_shape=(feature_length,)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(size_edge_label, use_bias=True, bias_initializer='glorot_uniform'),
    ], name='feature_scoring_model')


def ScoringModel(size_bi_lstm, size_feed_forward, size_edge_label, max_candidate_count, disable_handcrafted_features):
    scores_per_child = max_candidate_count * size_edge_label

    # Inputs
    bi_lstm_output = tf.keras.Input(shape=(None, size_bi_lstm), dtype=tf.float32, name='bi_lstm_output')
    candidates = tf.keras.Input(shape=(None, max_candidate_count, candidate_array_length),
                                dtype=tf.int32, name='candidates')
    if disable_handcrafted_features:
        numeric_features = None
    else:
        numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32,
                                      name='numeric_features')
    attended_nodes = tf.keras.Input(shape=(None, size_bi_lstm), dtype=tf.float32, name='attended_nodes')

    # Layers and models
    feature_model = FeatureModel(size_bi_lstm, disable_handcrafted_features)
    flatten_candidates = tf.keras.layers.Reshape(target_shape=(-1, candidate_array_length))
    feature_scoring_model = FeatureScoringModel(size_feed_forward, size_edge_label,
                                                get_feature_length(size_bi_lstm, disable_handcrafted_features))
    group_child_scores = tf.keras.layers.Reshape(target_shape=(-1, scores_per_child))
    softmax = tf.keras.layers.Softmax()

    # Assemble pieces
    # Flatten the child/parent candidates so that they are no longer grouped by child
    flat_candidates = flatten_candidates(candidates)
    if disable_handcrafted_features:
        feature_inputs = [bi_lstm_output, flat_candidates, attended_nodes]
    else:
        feature_inputs = [bi_lstm_output, flat_candidates, numeric_features, attended_nodes]
    flat_features = feature_model(feature_inputs)
    flat_scores = feature_scoring_model(flat_features)

    # Group them back together
    child_scores = group_child_scores(flat_scores)
    softmaxed_scores = softmax(child_scores)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [bi_lstm_output, candidates, attended_nodes]
    else:
        inputs = [bi_lstm_output, candidates, numeric_features, attended_nodes]
    return tf.keras.Model(inputs=inputs, outputs=[softmaxed_scores], name='scoring_model')

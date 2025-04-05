import tensorflow as tf

from shared.numeric_features import numeric_feature_length

candidate_array_length = 6


def get_feature_length(size_bi_lstm, disable_handcrafted_features):
    # See FeatureLayer.get_features_with_attention
    bi_lstm_length = 6 * size_bi_lstm
    if disable_handcrafted_features:
        return bi_lstm_length
    else:
        return bi_lstm_length + numeric_feature_length


class FeatureLayer(tf.keras.layers.Layer):
    def __init__(self, size_bi_lstm, disable_handcrafted_features, **kwargs):
        super(FeatureLayer, self).__init__(**kwargs)

        self.size_bi_lstm = size_bi_lstm
        self.disable_handcrafted_features = disable_handcrafted_features

    def get_config(self):
        config = super(FeatureLayer, self).get_config()
        config.update({
            'size_bi_lstm': self.size_bi_lstm,
            'disable_handcrafted_features': self.disable_handcrafted_features,
        })
        return config

    def call(self, inputs, **kwargs):
        """
        Expect inputs to be a tuple/list of
        (candidates, candidate_numeric_features, stacked_bi_lstm_output, stacked_attended_nodes)
        or
        (candidates, stacked_bi_lstm_output, stacked_attended_nodes)
        if disable_handcrafted_features is set to True
        """

        # Only process first batch to avoid GPU resource errors (we know batch size always = 1)
        # See https://github.com/tensorflow/tensorflow/issues/22094
        if self.disable_handcrafted_features:
            features = tf.map_fn(
                lambda t: self.get_features_with_attention_no_numeric(t[0], t[1], t[2]),
                # Second index 0 corresponds to batch
                [inputs[0][0], inputs[1][0], inputs[2][0]], dtype=tf.float32)
        else:
            features = tf.map_fn(
                lambda t: self.get_features_with_attention(t[0], t[1], t[2], t[3]),
                # Second index 0 corresponds to batch
                [inputs[0][0], inputs[1][0], inputs[2][0], inputs[3][0]], dtype=tf.float32)
        return tf.expand_dims(features, axis=0)

    def get_features_with_attention(self, candidate, numeric_features, bi_lstm_output,
                                    attended_nodes):
        # Node list starts with root node and padding node which have node_index -2, -1, so add 2
        # For candidate indices see get_candidate_array()
        parent_node_index = tf.add(candidate[0], 2)
        child_node_index = tf.add(candidate[3], 2)

        parent_attended = attended_nodes[parent_node_index]
        child_attended = attended_nodes[child_node_index]

        return tf.concat([
            # First and last words are generally likely to be important, so add them as features
            bi_lstm_output[candidate[1]],
            bi_lstm_output[candidate[2]],
            bi_lstm_output[candidate[4]],
            bi_lstm_output[candidate[5]],
            parent_attended,
            child_attended,
            numeric_features,
        ], axis=0)

    def get_features_with_attention_no_numeric(self, candidate, bi_lstm_output, attended_nodes):
        # Node list starts with root node and padding node which have node_index -2, -1, so add 2
        # For candidate indices see get_candidate_array()
        parent_node_index = tf.add(candidate[0], 2)
        child_node_index = tf.add(candidate[3], 2)

        parent_attended = attended_nodes[parent_node_index]
        child_attended = attended_nodes[child_node_index]

        return tf.concat([
            # First and last words are generally likely to be important, so add them as features
            bi_lstm_output[candidate[1]],
            bi_lstm_output[candidate[2]],
            bi_lstm_output[candidate[4]],
            bi_lstm_output[candidate[5]],
            parent_attended,
            child_attended,
        ], axis=0)

    def compute_output_shape(self, input_shape):
        candidate_input_shape = input_shape[0]  # input_shape is a list because this takes a list of inputs
        batch_size = candidate_input_shape[0]
        candidate_count = candidate_input_shape[1]
        return tf.TensorShape((batch_size, candidate_count,
                               get_feature_length(self.size_bi_lstm, self.disable_handcrafted_features)))


def FeatureModel(size_bi_lstm, disable_handcrafted_features):
    # Inputs
    batched_bi_lstm_output = tf.keras.Input(shape=(None, size_bi_lstm), dtype=tf.float32, name='bi_lstm_output')
    candidates = tf.keras.Input(shape=(None, candidate_array_length),
                                dtype=tf.int32, name='candidates')
    candidate_numeric_features = tf.keras.Input(shape=(None, numeric_feature_length), dtype=tf.float32,
                                                name='numeric_features')
    batched_attended_nodes = tf.keras.Input(shape=(None, size_bi_lstm), dtype=tf.float32, name='attended_nodes')

    # Functions for layers
    def repeat_bi_lstm_output(args):
        batched_candidates = args[0]
        batched_bi_lstm_output = args[1]

        batch_size = tf.shape(batched_bi_lstm_output)[0]
        bi_lstm_length = tf.shape(batched_bi_lstm_output)[1]
        candidate_count = tf.shape(batched_candidates)[1]
        return tf.reshape(tf.tile(batched_bi_lstm_output, [1, candidate_count, 1]),
                          (batch_size, candidate_count, bi_lstm_length, size_bi_lstm))

    def repeat_attended_nodes(args):
        batched_candidates = args[0]
        batched_attended_nodes = args[1]

        batch_size = tf.shape(batched_attended_nodes)[0]
        nodes_length = tf.shape(batched_attended_nodes)[1]
        candidate_count = tf.shape(batched_candidates)[1]
        return tf.reshape(tf.tile(batched_attended_nodes, [1, candidate_count, 1]),
                          (batch_size, candidate_count, nodes_length, size_bi_lstm))

    repeat_bilstm_lambda = tf.keras.layers.Lambda(repeat_bi_lstm_output, output_shape=(None, None, size_bi_lstm),
                                                  name='repeat_bi_lstm_output')
    repeat_attended_lambda = tf.keras.layers.Lambda(repeat_attended_nodes, output_shape=(None, None, size_bi_lstm),
                                                    name='repeat_attended_nodes')

    # Layers
    feature_layer = FeatureLayer(size_bi_lstm, disable_handcrafted_features)

    # Assemble pieces
    stacked_bi_lstm_output = repeat_bilstm_lambda([candidates, batched_bi_lstm_output])
    stacked_attended_nodes = repeat_attended_lambda([candidates, batched_attended_nodes])
    if disable_handcrafted_features:
        feature_layer_inputs = [candidates, stacked_bi_lstm_output, stacked_attended_nodes]
    else:
        feature_layer_inputs = [candidates, candidate_numeric_features, stacked_bi_lstm_output, stacked_attended_nodes]
    features = feature_layer(feature_layer_inputs)

    # Use Functional API to handle multiple inputs
    if disable_handcrafted_features:
        inputs = [batched_bi_lstm_output, candidates, batched_attended_nodes]
    else:
        inputs = [batched_bi_lstm_output, candidates, candidate_numeric_features,
                  batched_attended_nodes]
    return tf.keras.Model(inputs=inputs, outputs=[features], name='feature_model')

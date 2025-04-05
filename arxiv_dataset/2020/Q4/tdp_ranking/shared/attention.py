import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, size_bi_lstm, max_words_per_node, **kwargs):
        super(AttentionLayer, self).__init__(input_shape=(None, max_words_per_node, size_bi_lstm), **kwargs)

        self.size_bi_lstm = size_bi_lstm
        self.max_words_per_node = max_words_per_node

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.attention_weights = self.add_weight('attention_weights', shape=(self.size_bi_lstm, 1),
                                                 initializer=tf.keras.initializers.glorot_uniform())

        super(AttentionLayer, self).build(input_shape)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'size_bi_lstm': self.size_bi_lstm,
            'max_words_per_node': self.max_words_per_node,
        })
        return config

    def call(self, inputs, **kwargs):
        """
        Given a matrix containing the Bi-LSTM outputs for each word in a node, return a weighted sum of the outputs for
        those words, as determined by attending to the (outputs for the) words.
        Takes a batch of lists of nodes (4D tensor).
        """

        # Make copy of attention_to_words for each node (TF 1.13 doesn't broadcast matrix multiplication)
        batch_size = tf.shape(inputs)[0]
        node_count = tf.shape(inputs)[1]
        attention_weights_stacked = tf.reshape(tf.tile(self.attention_weights, (batch_size * node_count, 1)),
                                               shape=(batch_size, node_count, self.size_bi_lstm, 1))
        attention_to_words = tf.matmul(inputs, attention_weights_stacked)

        # Interpret negative weights as no attention
        unnormalized_weights_for_words = tf.nn.relu(attention_to_words)

        # Axis 0: batch, axis 1: nodes, axis 2: words, axis 3: weights (1-dim)
        # n.B. softmax is diluting signal of actual words because softmaxing on padding (zeros) as well
        weights_for_words = tf.nn.softmax(unnormalized_weights_for_words, axis=2)

        # Transpose word weights per node (do not transpose 2 outermost batch dimensions)
        transposed_weights = tf.transpose(weights_for_words, perm=[0, 1, 3, 2])
        weighted_sums = tf.matmul(transposed_weights, inputs)
        # Remove size 1 dim in axis 2 (formerly words dimension) to make it a vector per node
        return weighted_sums[:, :, 0, :]

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        node_count = input_shape[1]
        return tf.TensorShape((batch_size, node_count, self.size_bi_lstm))


class NodeToWordsLayer(tf.keras.layers.Layer):
    def __init__(self, size_bi_lstm, max_words_per_node, **kwargs):
        super(NodeToWordsLayer, self).__init__(**kwargs)

        self.size_bi_lstm = size_bi_lstm
        self.max_words_per_node = max_words_per_node

    def get_config(self):
        config = super(NodeToWordsLayer, self).get_config()
        config.update({
            'size_bi_lstm': self.size_bi_lstm,
            'max_words_per_node': self.max_words_per_node,
        })
        return config

    def call(self, inputs, **kwargs):
        """Expect inputs to be a tuple/list of batched_nodes, batched_bi_lstm_outputs"""

        # Only process first batch to avoid GPU resource errors (we know batch size always = 1)
        # See https://github.com/tensorflow/tensorflow/issues/22094
        batch_0 = [inputs[0][0], inputs[1][0]]
        return tf.expand_dims(self.get_words_for_nodes(batch_0), axis=0)

    def get_words_for_nodes(self, nodes_inputs):
        return tf.map_fn(lambda t: self.get_words_for_node(t[0], t[1]), nodes_inputs, dtype=tf.float32)

    def get_words_for_node(self, node, bi_lstm_output):
        """
        If node is a normal node, return the bi_lstm embeddings for each word in the node.

        If node is a pre-defined meta node, then self.start_word_index and self.end_word_index are -1: use the last
        word (whole document) to represent it and return the bi_lstm embeddings for that.
        This applies to both the root node and the past, present and future nodes (but not DCT).
        """
        start_word_index = node[0]
        end_word_index = node[1]
        word_count = tf.add(tf.subtract(end_word_index, start_word_index), 1)

        words = tf.cond(tf.equal(end_word_index, -1),
                        lambda: tf.expand_dims(bi_lstm_output[start_word_index], axis=0),
                        # TODO does this really always return a slice and not an index?
                        lambda: bi_lstm_output[start_word_index:tf.add(end_word_index, 1)])
        # Help TensorFlow understand that this always results in the same shape
        words.set_shape((None, self.size_bi_lstm))

        return self.pad(words, word_count)

    def pad(self, words, word_count):
        padding = tf.subtract(self.max_words_per_node, word_count)
        padded_words = tf.case({
            tf.greater(padding, 0): lambda: tf.pad(words, [[0, padding], [0, 0]]),
            tf.less(padding, 0): lambda: words[:self.max_words_per_node]
        },
            default=lambda: words,
            exclusive=True)
        # Help TensorFlow understand that this always results in the same shape
        padded_words.set_shape((self.max_words_per_node, self.size_bi_lstm))
        return padded_words

    def compute_output_shape(self, input_shape):
        batched_nodes_shape = input_shape[0]  # input_shape is a list because this takes a list of inputs
        batch_size = batched_nodes_shape[0]
        node_count = batched_nodes_shape[1]
        return tf.TensorShape((batch_size, node_count, self.max_words_per_node, self.size_bi_lstm))


def AttentionModel(size_bi_lstm, max_words_per_node):
    # Inputs
    batched_nodes = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name='nodes')
    batched_bi_lstm_output = tf.keras.Input(shape=(None, size_bi_lstm), dtype=tf.float32, name='bi_lstm_output')

    # Functions for layers
    def repeat_bi_lstm_output(args):
        batched_nodes = args[0]
        batched_bi_lstm_output = args[1]

        batch_size = tf.shape(batched_bi_lstm_output)[0]
        bi_lstm_length = tf.shape(batched_bi_lstm_output)[1]
        node_count = tf.shape(batched_nodes)[1]
        return tf.reshape(tf.tile(batched_bi_lstm_output, [1, node_count, 1]),
                          (batch_size, node_count, bi_lstm_length, size_bi_lstm))

    # Layers
    repeat_bilstm_lambda = tf.keras.layers.Lambda(repeat_bi_lstm_output, output_shape=(None, None, size_bi_lstm),
                                                  name='repeat_bi_lstm_output')
    nodes_to_words_layer = NodeToWordsLayer(size_bi_lstm, max_words_per_node)
    attention_layer = AttentionLayer(size_bi_lstm, max_words_per_node)

    # Assemble pieces
    stacked_bi_lstm_output = repeat_bilstm_lambda([batched_nodes, batched_bi_lstm_output])
    batched_words_for_nodes = nodes_to_words_layer([batched_nodes, stacked_bi_lstm_output])
    attended_nodes = attention_layer(batched_words_for_nodes)

    # Use Functional API to handle multiple inputs
    return tf.keras.Model(inputs=[batched_nodes, batched_bi_lstm_output], outputs=[attended_nodes],
                          name='attention_model')


def validate_node_words(nodes, max_words_per_node):
    for node in nodes:
        word_count = node.end_word_index_in_doc - node.start_word_index_in_doc + 1
        if word_count > max_words_per_node:
            print('WARNING: Node {} (\'{}\') exceeds max words per node, will be truncated'
                  .format(node.ID, node.words))

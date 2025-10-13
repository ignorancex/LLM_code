
import tensorflow as tf

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super(GraphAttentionLayer, self).__init__()
        self.out_dim = out_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", (input_shape[0][-1], self.out_dim))
        self.attention_kernel = self.add_weight("attention_kernel", (2*self.out_dim, 1))

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        node_features_transformed = tf.keras.activations.relu(tf.matmul(node_features, self.kernel))
        
        a_input = tf.concat([
            tf.tile(tf.expand_dims(node_features_transformed, 1), [1, tf.shape(node_features_transformed)[1], 1, 1]), 
            tf.tile(tf.expand_dims(node_features_transformed, 2), [1, 1, tf.shape(node_features_transformed)[1], 1])
        ], axis=-1)
        
        e = tf.keras.activations.relu(tf.matmul(a_input, self.attention_kernel))
        
        # Make sure adjacency_matrix is in the right shape
        adjacency_matrix = tf.expand_dims(adjacency_matrix, axis=-1)
        
        attention_weights = tf.nn.softmax(tf.multiply(adjacency_matrix, e), axis=2)
        
        output = tf.matmul(attention_weights, node_features_transformed)
        
        return output

class GAT(tf.keras.Model):
    def __init__(self, num_classes):
        super(GAT, self).__init__()
        self.graph_attention_1 = GraphAttentionLayer(64)
        self.graph_attention_2 = GraphAttentionLayer(32)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        x = self.graph_attention_1([adjacency_matrix, node_features])
        x = self.graph_attention_2([adjacency_matrix, x])
        x = tf.reduce_mean(x, axis=1)
        x = self.fc(x)
        return x

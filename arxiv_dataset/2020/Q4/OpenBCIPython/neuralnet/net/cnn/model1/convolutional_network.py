
import numpy as np
import tensorflow as tf
import json

class CNNModel1:
    def __init__(self, project_dir, meta_info_data_set):

        meta_info_file = project_dir + "/neuralnet/net/cnn/model1/config.json"

        with open(meta_info_file) as data_file:
            meta_info = json.load(data_file)
            self.learning_rate = float(meta_info["net"]["learning_rate"])
            self.conv1_features = int(meta_info["net"]["conv1_features"])
            self.conv2_features = int(meta_info["net"]["conv2_features"])
            self.max_pool_size1 = int(meta_info["net"]["max_pool_size1"])
            self.max_pool_size2 = int(meta_info["net"]["max_pool_size2"])
            self.fully_connected_size1 = int(meta_info["net"]["fully_connected_size1"])
            self.filter_side = int(meta_info["net"]["filter_side"])
            self.dropout = int(meta_info["net"]["dropout"])
            self.training_iters = int(meta_info["net"]["training_iters"])
            self.display_step = int(meta_info["net"]["display_step"])
            self.strides_layer1 = int(meta_info["net"]["strides_layer1"])
            self.strides_layer2 = int(meta_info["net"]["strides_layer2"])
            self.model_path = project_dir + str(meta_info["net"]["model_path"])
            self.logs_path = project_dir + str(meta_info["net"]["logs_path"])

            self.num_channels = int(meta_info_data_set["num_channels"])
            self.n_classes = int(meta_info_data_set["number_of_class"])
            self.image_width = int(meta_info_data_set["generated_image_width"])
            self.image_height = int(meta_info_data_set["generated_image_height"])
            self.feature_vector_size = self.image_height*self.image_width

        with tf.name_scope("Dropout"):
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('Weights'):
        # Store layers weight & bias
            self.weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([self.filter_side, self.filter_side, self.num_channels, self.conv1_features]), name='Weights_wc1'),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([self.filter_side, self.filter_side, self.conv1_features, self.conv2_features]), name='Weights_wc2'),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([3*3*self.conv2_features, self.fully_connected_size1]), name='Weights_wd1'),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd2': tf.Variable(tf.random_normal([self.fully_connected_size1, self.fully_connected_size1]), name='Weights_wd2'),
                # 1024 inputs, 10 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([self.fully_connected_size1, self.n_classes]), name='Weights_out')
            }

        with tf.name_scope('Biases'):

            self.biases = {
                'bc1': tf.Variable(tf.random_normal([self.conv1_features]), name='bc1'),
                'bc2': tf.Variable(tf.random_normal([self.conv2_features]), name='bc2'),
                'bd1': tf.Variable(tf.random_normal([self.fully_connected_size1]), name='bd1'),
                'bd2': tf.Variable(tf.random_normal([self.fully_connected_size1]), name='bd2'),
                'out': tf.Variable(tf.random_normal([self.n_classes]), name='out')
            }

        # Declare model placeholders
        # x_input_shape = (batch_size, image_width, image_height, num_channels)
        with tf.name_scope('Inputs'):
            self.x_input = tf.placeholder(tf.float32, [None, self.feature_vector_size], name='InputData')
            self.y_target = tf.placeholder(tf.int32, [None, self.n_classes], name='LabelData')

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create accuracy function
    def get_accuracy(self, logits, targets):
        batch_predictions = np.argmax(logits, axis=1)
        num_correct = np.sum(np.equal(batch_predictions, targets))
        return (100. * num_correct / batch_predictions.shape[0])

    # Initialize Model Operations
    def conv_net(self, input_data):

        input_data = tf.reshape(input_data, shape=[-1, self.image_width, self.image_height, self.num_channels])

        # First Conv-ReLU-MaxPool Layer
        result_of_first_cnn_layer = self.conv2d(input_data, self.weights["wc1"], self.biases["bc1"], strides=self.strides_layer1)
        tf.summary.histogram("result_of_first_cnn_layer", result_of_first_cnn_layer)
        result_of_first_max_polling_layer = self.maxpool2d(result_of_first_cnn_layer, k=2)

        # Second Conv-ReLU-MaxPool Layer
        result_of_second_cnn_layer = self.conv2d(result_of_first_max_polling_layer, self.weights["wc2"], self.biases["bc2"], strides=self.strides_layer1)
        tf.summary.histogram("result_of_second_cnn_layer", result_of_second_cnn_layer)
        result_of_second_max_polling_layer = self.maxpool2d(result_of_second_cnn_layer, k=2)

        # Transform Output into a 1xN layer for next fully connected layer
        final_conv_shape = result_of_second_max_polling_layer.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(result_of_second_max_polling_layer, [-1 , final_shape])

        # First Fully Connected Layer
        fully_connected1 = tf.add(tf.matmul(flat_output, self.weights["wd1"]), self.biases["bd1"])
        fully_connected1 = tf.nn.relu(fully_connected1)
        tf.summary.histogram("fully_connected1", fully_connected1)

        # Apply Dropout
        fully_connected1 = tf.nn.dropout(fully_connected1, self.keep_prob)

        # Second Fully Connected Layer
        fully_connected2 = tf.add(tf.matmul(fully_connected1, self.weights["wd2"]), self.biases["bd2"])
        fully_connected2 = tf.nn.relu(fully_connected2)
        tf.summary.histogram("fully_connected2", fully_connected2)

        # Output, class prediction
        out = tf.add(tf.matmul(fully_connected2, self.weights['out']), self.biases['out'])

        return out,None

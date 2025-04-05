import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

import numpy as np


IMAGE_SIZE = 28

def get_conv_dimension(filter_list):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Model function for CNN."""
            features = tf.placeholder(
                tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
            labels = tf.placeholder(tf.int64, shape=[None], name='labels')
            input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            conv1 = tf.layers.conv2d(
              inputs=input_layer,
              filters=filter_list[0],
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=filter_list[1],
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
#     return int(np.prod(pool2.get_shape().as_list()[1:]))
    return pool2.get_shape().as_list()

if __name__ == "__main__":
    tf.autograph.set_verbosity(0)
    print(get_conv_dimension([32, 64]))
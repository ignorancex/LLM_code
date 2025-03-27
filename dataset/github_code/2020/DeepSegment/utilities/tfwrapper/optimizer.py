# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import tensorflow as tf


def create_optimizer_from_string(optimizer_str):
    optimizer_str = str(optimizer_str).lower()

    if optimizer_str in ["adam"]:
        return tf.train.AdamOptimizer
    elif optimizer_str in ["rmsprop", "rms-prop"]:
        return tf.train.RMSPropOptimizer
    elif optimizer_str in ["sgd", "stochasticgradient", "stochastic-gradient"]:
        return tf.keras.optimizers.SGD
    else:
        raise ValueError("Unknown optimizer '{:}'".format(optimizer_str))

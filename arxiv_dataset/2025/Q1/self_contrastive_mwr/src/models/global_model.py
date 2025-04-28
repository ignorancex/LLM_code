from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import custom_layers


def get_model(input_shape: Tuple[int], num_classes: int, is_contrastive: bool = False) -> tf.keras.Model:
    """
    Returns the global model (either contrastive or non-contrastive variant).

    Args:
        input_shape (Tuple[int]): The shape of the input tensor.
        num_classes (int): The number of output classes.
        is_contrastive (bool, optional): Whether the model uses contrastive learning. Defaults to False.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    x_input = keras.Input(shape=input_shape, name='global_input')
    x_flipped_input = keras.Input(shape=input_shape, name='global_input_flipped')

    siamese_model = custom_layers.shared_layer(input_shape, units=256, num_layers=4,
                                               prefix='global_shared')
    x = siamese_model(x_input)
    # Do not need to normalize as we expect them to be the same (close) if they are healthy
    #x = tf.math.l2_normalize(x, axis=-1)
    x = custom_layers.Identity(name='global_shared_output')(x)
    xf = siamese_model(x_flipped_input)
    #xf = tf.math.l2_normalize(xf, axis=-1)
    xf = custom_layers.Identity(name='global_shared_output_flipped')(xf)
    
    if is_contrastive:
        x_loss = custom_layers.Identity(name='global_contrastive')(x)
    
    x = tf.abs(x - xf)
    # Convolution of size 1 to act as a low pass difference filter
    x = tf.expand_dims(x, axis=-1)
    x = layers.Conv1D(1, 1, activation='relu')(x)
    x = tf.squeeze(x, axis=-1)

    x = tf.reduce_sum(x, axis=-1, keepdims=True, name='global_diff')
    
    x_output = layers.Dense(num_classes, activation='tanh',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            name='global_output')(x)
    
    if is_contrastive:
        x_outputs = [x_loss, x_output]
    else:
        x_outputs = [x_output]
    
    model = keras.Model([x_input, x_flipped_input], x_outputs)
    
    return model



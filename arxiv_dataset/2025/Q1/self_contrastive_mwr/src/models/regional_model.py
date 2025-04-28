from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import custom_layers


def get_model(input_shape: Tuple[int], num_classes: int,
              is_contrastive: bool = False) -> tf.keras.Model:
    """
    Returns the regional model (either contrastive or non-contrastive variant).

    Args:
        input_shape (Tuple[int]): The shape of the input data.
        num_classes (int): The number of output classes.
        is_contrastive (bool, optional): Whether the model uses contrastive learning. Defaults to False.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    xl_input = keras.Input(shape=input_shape, name='regional_input_l')
    xr_input = keras.Input(shape=input_shape, name='regional_input_r')
    
    siamese_model = custom_layers.shared_layer(input_shape, units=256, num_layers=4,
                                               prefix='regional_shared')
    
    xl = siamese_model(xl_input)
    xl = tf.math.l2_normalize(xl, axis=-1, epsilon=1e-9)
    xl = custom_layers.Identity(name='regional_shared_output_l')(xl)
    
    xr = siamese_model(xr_input)
    xr = tf.math.l2_normalize(xr, axis=-1, epsilon=1e-9)
    xr = custom_layers.Identity(name='regional_shared_output_r')(xr)

    if is_contrastive:
        x_loss = tf.math.add(xl, xr)
        x_loss = custom_layers.Identity(name='regional_contrastive')(x_loss)

    x = tf.abs(xl - xr)
    # Convolution of size 1 to act as a low pass difference filter
    x = tf.expand_dims(x, axis=-1)
    x = layers.Conv1D(1, 1, activation='relu')(x)
    x = tf.squeeze(x, axis=-1)

    x = tf.reduce_sum(x, axis=-1, keepdims=True, name='regional_diff')
    
    x_output = layers.Dense(num_classes, activation='tanh',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            name='regional_output')(x)
    
    if is_contrastive:
        x_outputs = [x_loss, x_output]
    else:
        x_outputs = [x_output]
    
    model = keras.Model([xl_input, xr_input], x_outputs)
    
    return model



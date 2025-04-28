from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import custom_layers


def get_model(input_shape: Tuple[int], num_of_points: int, num_classes: int,
              is_contrastive: bool = False) -> tf.keras.Model:
    """
    Returns the local model (either contrastive or non-contrastive variant).

    Args:
        input_shape (Tuple[int]): The shape of the input tensor.
        num_of_points (int): The number of points usd in the network.
        num_classes (int): The number of output classes.
        is_contrastive (bool, optional): Whether the model uses contrastive learning. Defaults to False.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    x_inputs = []
    
    point_model = custom_layers.shared_layer(input_shape, units=64, final_units=1,
                                             num_layers=4, prefix='local_shared')
    
    x = []
    for i in range(num_of_points):
        x_input = keras.Input(shape=input_shape, name=f'local_input_{i}')
        x_inputs.append(x_input)
        x.append(point_model(x_input))
            
    x = layers.Concatenate(axis=-1, name='local_output_individual')(x)

    if is_contrastive:
        x_loss = tf.math.l2_normalize(x, axis=-1, epsilon=1e-9)
        x_loss = custom_layers.Identity(name='local_contrastive')(x_loss)

    # b x n x n
    x = tf.math.abs(tf.expand_dims(x, axis=2) - tf.expand_dims(x, axis=1))
    # b x n x n x 1
    x = tf.expand_dims(x, axis=-1)
    # Convolution of size 1 to act as a low pass difference filter
    x = layers.Conv2D(1, 1, activation='relu')(x)
    # b x n x n
    x = tf.squeeze(x, axis=-1)  
    # b
    x = tf.math.reduce_mean(x, axis=[1,2], keepdims=False)
    # b x 1
    x = tf.expand_dims(x, axis=-1)
    
    x_output = layers.Dense(num_classes, activation='tanh',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            name='local_output')(x)
    
    if is_contrastive:
        x_outputs = [x_loss, x_output]
    else:
        x_outputs = [x_output]
        
    model = keras.Model(x_inputs, x_outputs)
    
    return model


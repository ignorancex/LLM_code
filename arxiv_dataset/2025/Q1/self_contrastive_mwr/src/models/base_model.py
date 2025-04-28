from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import custom_layers


def get_model(input_shape: Tuple[int], num_classes: int,
              is_contrastive: bool = False) -> keras.Model:
    """
    Creates the baseline model for classification with or without contrastive learning.

    Args:
        input_shape (Tuple[int]): The shape of the input data.
        num_classes (int): The number of classes for classification.
        is_contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.

    Returns:
        keras.Model: The constructed Keras model.
    """
    core_model = custom_layers.shared_layer(input_shape, units=256, num_layers=4,
                                            final_layer=False, prefix='base_shared')
    
    x_input = keras.Input(shape=input_shape, name='base_input')
    x = core_model(x_input)
    
    if is_contrastive:
        x_loss = custom_layers.Identity(name='base_contrastive')(tf.math.l2_normalize(x, axis=-1))
    
    x_output = layers.Dense(num_classes,
                            activation='sigmoid',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            name='base_output')(x)
    
    if is_contrastive:
        x_outputs = [x_loss, x_output]
    else:
        x_outputs = [x_output]
    
    model = keras.Model(x_input, x_outputs)
    
    return model
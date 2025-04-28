from typing import Optional, Tuple

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Identity")
class Identity(layers.Layer):
    """
    Identity layer.
    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.
    Args:
        name: Optional name for the layer instance.
    """
    
    def __init__(self, **kwargs):
            """
            Initialize the Identity layer.

            Args:
                **kwargs: Additional keyword arguments.

            Returns:
                None
            """
            super(Identity, self).__init__(**kwargs)
        
        
    def get_config(self):
            """
            Returns the configuration dictionary for the layer.

            Returns:
                dict: The configuration dictionary for the layer.
            """
            config = super().get_config()
            return config
    
    
    def call(self, inputs):
            """
            Applies the identify to the given inputs.

            Args:
                inputs (Tensor): The input tensor.

            Returns:
                Tensor: The output tensor after applying the identity operation.
            """
            return tf.identity(inputs)
    
    
    
def shared_layer(input_shape: Tuple[int], units: int = 256, final_units: Optional[int] = None,
                 num_layers: int = 4, final_activation: str = 'relu',
                 final_layer: bool = True, prefix: str = 'shared_layer') -> tf.keras.Model:
    """
    Create a sub-model used across the proposed models as part of their Simaese layers.

    Args:
        input_shape (Tuple[int]): The shape of the input tensor.
        units (int, optional): The number of units in each dense layer. Defaults to 256.
        final_units (int, optional): The number of units in the final dense layer. Defaults to None.
        num_layers (int, optional): The number of dense layers in the model. Defaults to 4.
        final_activation (str, optional): The activation function for the final dense layer. Defaults to 'relu'.
        final_layer (bool, optional): Whether to include a final dense layer. Defaults to True.
        prefix (str, optional): The prefix for the names of the layers. Defaults to 'shared_layer'.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    x_input = layers.Input(shape=input_shape, name=f'{prefix}_input')
    x = x_input
    
    prev_x = None

    for i in range(num_layers):
        x = layers.Dense(units,
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name=f'{prefix}_dense_1_{i:03d}')(x)
        
        x = layers.LayerNormalization(axis=-1, center=True, scale=True,
                                      name=f'{prefix}_norm_1_{i:03d}')(x)
        
        x = layers.Activation('relu', name=f'{prefix}_activation_1_{i:03d}')(x)
        
        x = layers.Dense(units,
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name=f'{prefix}_dense_2_{i:03d}')(x)
        
        x = layers.LayerNormalization(axis=-1, center=True, scale=True,
                                      name=f'{prefix}_norm_2_{i:03d}')(x)
        
        if not prev_x is None:
            x = layers.Add(name=f'{prefix}_add_{i:03d}')([x, prev_x])
        
        x = layers.Activation('relu', name=f'{prefix}_activation_2_{i:03d}')(x)
        prev_x = x
                
    if final_units is None:
        final_units = units
    if final_layer:
        x_output = layers.Dense(final_units, activation=final_activation,
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                name=f'{prefix}_output')(x)
    else:
        x_output = x

    return keras.Model(inputs=[x_input], outputs=[x_output])
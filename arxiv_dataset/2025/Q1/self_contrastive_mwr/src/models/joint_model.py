from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import local_model, regional_model, global_model, custom_layers


def get_model(input_shapes: Dict[str,Tuple[int]], num_of_points: int,
              num_classes: int, is_contrastive: bool = False) -> tf.keras.Model:
    """
    Returns the joint model (either contrastive or non-contrastive variant).

    Args:
        input_shapes (Dict[str,Tuple[int]]): A dictionary specifying the shapes of input tensors for different subnetworks.
            The keys are 'local', 'regional', and 'global', and the values are tuples representing the shape of each input tensor.
        num_of_points (int): The number of points usd in the local subnetwork.
        num_classes (int): The number of output classes.
        is_contrastive (bool, optional): Whether the model uses contrastive learning. Defaults to False.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    # Prepare subnetworks' inputs
    x_local_inputs = []
    for i in range(num_of_points):
        x_local_inputs.append(keras.Input(shape=input_shapes['local'], name=f'joint_local_input_{i}'))
        
    xl_regional_input = keras.Input(shape=input_shapes['regional'], name='joint_regional_input_l')
    xr_regional_input = keras.Input(shape=input_shapes['regional'], name='joint_regional_input_r')
    
    x_global_input = keras.Input(shape=input_shapes['global'], name='joint_global_input')
    x_global_flipped_input = keras.Input(shape=input_shapes['global'], name='joint_global_input_flipped')
    
    # Prepare subnetworks
    local_network = local_model.get_model(input_shapes['local'], num_of_points,
                                          num_classes, is_contrastive)
    regional_network = regional_model.get_model(input_shapes['regional'],
                                                num_classes, is_contrastive)
    global_network = global_model.get_model(input_shapes['global'],
                                            num_classes, is_contrastive)
    
    
    x_local_output = local_network(x_local_inputs)
    x_regional_output = regional_network([xl_regional_input, xr_regional_input])
    x_global_output = global_network([x_global_input, x_global_flipped_input])
    
    if is_contrastive:
        # Select the output from each of the models
        x_results = {name: output for name, output in zip(local_network.output_names, x_local_output)}
        x_local_output = x_results['local_output']
        x_local_loss = x_results['local_contrastive']
        
        x_results = {name: output for name, output in zip(regional_network.output_names, x_regional_output)}
        x_regional_output = x_results['regional_output']
        x_regional_loss = x_results['regional_contrastive']
        
        x_results = {name: output for name, output in zip(global_network.output_names, x_global_output)}
        x_global_output = x_results['global_output']
        x_global_loss = x_results['global_contrastive']
        
        x_local_loss = custom_layers.Identity(name='joint_local_contrastive')(x_local_loss)
        x_regional_loss = custom_layers.Identity(name='joint_regional_contrastive')(x_regional_loss)
        x_global_loss = custom_layers.Identity(name='joint_global_contrastive')(x_global_loss)
    
    x_local_loss_output = custom_layers.Identity(name='local_output')(x_local_output)
    x_regional_loss_output = custom_layers.Identity(name='regional_output')(x_regional_output)
    x_global_loss_output = custom_layers.Identity(name='global_output')(x_global_output)
    
    # Join individual outputs and make final prediction
    x_local_output = layers.Dense(1, activation='relu',
                                  use_bias=True,
                                  kernel_initializer='ones',
                                  bias_initializer='zeros')(x_local_output)
    x_regional_output = layers.Dense(1, activation='relu',
                                  use_bias=True,
                                  kernel_initializer='ones',
                                  bias_initializer='zeros')(x_regional_output)
    x_global_output = layers.Dense(1, activation='relu',
                                  use_bias=True,
                                  kernel_initializer='ones',
                                  bias_initializer='zeros')(x_global_output)
    
    x = layers.Concatenate(axis=-1, name='joint_concat')([x_local_output,
                                                          x_regional_output,
                                                          x_global_output])
    
    x_output = layers.Dense(num_classes, activation='tanh',
                            use_bias=True,
                            kernel_initializer='ones',
                            bias_initializer='zeros',
                            name='joint_output')(x)
    
    x_inputs = [*x_local_inputs, xl_regional_input, xr_regional_input,
                x_global_input, x_global_flipped_input]
    
    if is_contrastive:
        x_outputs = [x_local_loss, x_regional_loss, x_global_loss]
    else:
        x_outputs = []
    
    x_outputs.extend([x_local_loss_output, x_regional_loss_output, x_global_loss_output, x_output])
        
    model = keras.Model(x_inputs, x_outputs)
    
    return model
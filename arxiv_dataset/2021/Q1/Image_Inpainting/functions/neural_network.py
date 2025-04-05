"""Neural-network architecture for MNIST classification

The functions in this module allow to train a neural network based on dense layers

Author: Sergio P. Perez
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model


#####################################################################################
#
# FUNCTION: temporal_loop
#
#####################################################################################

def training(train_images, train_labels):
    """Training and definition of the neural network

    Args:
        train_images: training set of images
        train_labels: labels of the training images
    Returns: 
        model: tensorflow trained model object
        pd.DataFrame(history.history): training history
    """

    # Build the model
    model = Sequential([
      Dense(64, activation='relu', input_shape=(784,)), 
      Dense(64, activation='relu'),  
      Dense(10, activation='softmax'), 
    ])
    
    # Compile the model
    model.compile(
      optimizer='adam',    
      loss='categorical_crossentropy', 
      metrics=['accuracy'], 
    )
    
    # Train the model
    history = model.fit( 
      train_images,
      to_categorical(train_labels),
      epochs=8, 
      batch_size=32,  
    )
    
    return model, pd.DataFrame(history.history)



#####################################################################################
#
# CLASS: DenseLayer
#
#####################################################################################


class DenseLayer(Layer):
    """Subclass inherited from the Layer class of TensorFlow."""
    
    def __init__(self, units, input_dim):
        """Define weights and bias

        Args:
            units: number of neurons
            input_dim: input dimensions
        """
        super(DenseLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal')
        self.b = self.add_weight(shape=(units,),
                             initializer='zeros')
    def call(self, inputs):
        """Implement matrix multiplication

        Args:
            inputs: input of the dense layer
        Returns: 
            tf.matmul(inputs, self.w)+self.b: output of dense layer
        """
        return tf.matmul(inputs, self.w)+self.b

#####################################################################################
#
# CLASS: MyModel
#
#####################################################################################   

class MyModel(Model):
    """Subclass inherited from the Model class of TensorFlow."""

    def __init__(self, units_1, input_dim_1, units_2, units_3):
        """Define layers and softmax function

        Args:
            units_1: number of neurons in layer_1
            input_dim_1: input dimensions in layer_1
            units_2: number of neurons in layer_2
            units_3: number of neurons in layer_3
        """
        super(MyModel, self).__init__()
        self.layer_1 = DenseLayer(units_1, input_dim_1)
        self.layer_2 = DenseLayer(units_2, units_1)
        self.layer_3 = DenseLayer(units_3, units_2)
        self.softmax = Softmax()

    def call(self, inputs):
        """Define layers and softmax function

        Args:
            units_1: number of neurons in layer_1
            input_dim_1: input dimensions in layer_1
            units_2: number of neurons in layer_2
            units_3: number of neurons in layer_3
        """
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = self.layer_3(x)
        return self.softmax(x)

#####################################################################################
#
# FUNCTION: grad
#
##################################################################################### 
        
@tf.function # decorator to speed up customized training loop
def grad(model, inputs, targets):
    """Automatically compute gradients from loss function

        Args:
            model: Tensorflow model
            inputs: input of the neural network
            targets: training labels
        Returns: 
            loss_value: evaluation of the loss function
            tape.gradient(loss_value, model.trainable_variables): 
                gradients of the trainable weights
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#####################################################################################
#
# FUNCTION: loss
#
#####################################################################################
    
def loss(model, x, y):
    """Automatically compute gradients from loss function

        Args:
            model: Tensorflow model
            x: input of the neural network
            y: ground-truth labels
        Returns: 
            loss_object(y_true=y, y_pred=y_): loss function evaluation
    """
    y_ = model(x)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    return loss_object(y_true=y, y_pred=y_)


#####################################################################################
#
# FUNCTION: customized_training
#
#####################################################################################
    
def customized_training(train_images, train_labels):
    """Customized training loop

        Args:
            train_images: training set of images
            train_labels: training set of labels
        Returns: 
            model: trained Tersorflow model
    """
    
    model = MyModel(64, 784, 64, 10) # Instantiate model with MNIST input dimensions
    model(tf.ones((1, 784))) # Initialize model by passing an input to it
    model.summary() # Print model summary
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Select optimizer
    
    # Turn training images into tf dataset 
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.batch(32) # Divide data in batches
    
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    
    num_epochs = 5 # Select number of epochs
    
    # Measure the training time
    start_time = time.time()
    
    for epoch in range(num_epochs): # Set up epoch loop
        epoch_loss_avg = tf.keras.metrics.Mean() # Select how to average losses
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy() # Select accuracy
     
        for x, y in train_dataset: # Training loop
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
            # Compute current loss
            epoch_loss_avg(loss_value)  
            # Compare predicted label to actual label
            epoch_accuracy(to_categorical(y), model(x))
    
      # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
    
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                                epoch, epoch_loss_avg.result(),
                                epoch_accuracy.result()))
        
    print("Duration :{:.3f}".format(time.time() - start_time))
    
    return model # Return trained model

#####################################################################################
#
# FUNCTION: customized_validation
#
#####################################################################################
    
def customized_validation(test_images, test_labels, model):
    """Validate customized model

        Args:
            test_images: test set of images
            test_labels: test set of labels
            model: Tensorflow trained model
    """
    
    # Turn test images into tf dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(32) # Divide data in batches
    
    epoch_loss_avg = tf.keras.metrics.Mean() # Select how to average losses
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy() # Select accuracy
    
    # Test loop
    for x, y in test_dataset:
        # Optimize the model
        loss_value = loss(model, x, y)
        
        # Compute current loss
        epoch_loss_avg(loss_value)  
        # Compare predicted label to actual label
        epoch_accuracy(to_categorical(y), model(x))
    
    # End epoch
    print("Test loss: {:.3f}".format(epoch_loss_avg.result().numpy()))
    print("Test accuracy: {:.3%}".format(epoch_accuracy.result().numpy()))

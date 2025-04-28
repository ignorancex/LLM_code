# here we define the possible architectures
# number of layers, nodes etc are kept as input to the model class to allow keras tuner study
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Input, Model
from tensorflow.keras.constraints import MaxNorm

# a classifier
def class_model(n_lay, n_units, lr, decay_steps, decay_rate, bn):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=decay_steps, 
    decay_rate=decay_rate,
    staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    Classifier = tf.keras.Sequential()
    for _ in range(n_lay): # let's add N hidden layers
        Classifier.add(tf.keras.layers.Dense(n_units, kernel_constraint=MaxNorm(5)))
        if bn:
            Classifier.add(tf.keras.layers.BatchNormalization())
        Classifier.add(tf.keras.layers.PReLU())
    Classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    Classifier.compile( 
        optimizer=optimizer,
        loss="binary_crossentropy",
     metrics=["accuracy"],   
    )
    return Classifier

# a classifier with convolutional layers
def class_model_cnn(n_lay, n_units, lr, decay_steps, decay_rate, bn, filters, maxp):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=decay_steps, 
    decay_rate=decay_rate,
    staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    Classifier = tf.keras.Sequential()
    for n_filt in filters:
        Classifier.add(tf.keras.layers.Conv1D(filters = n_filt, kernel_size= 1, strides=1, activation='relu', kernel_initializer='he_uniform'))
    # flatten or pool?
    if maxp:
        Classifier.add(tf.keras.layers.MaxPooling1D(2))
    elif maxp ==0:
        Classifier.add(tf.keras.layers.Conv1D(filters = n_filt, kernel_size= 2, strides=1, activation='relu', kernel_initializer='he_uniform'))
    Classifier.add(tf.keras.layers.Flatten())
    for _ in range(n_lay): # let's add N hidden layers
        Classifier.add(tf.keras.layers.Dense(n_units, kernel_constraint=MaxNorm(5)))
        if bn:
            Classifier.add(tf.keras.layers.BatchNormalization())
        Classifier.add(tf.keras.layers.PReLU())
    Classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    

    Classifier.compile( 
        optimizer=optimizer,
        loss="binary_crossentropy",
     metrics=["accuracy"],   
    )
    return Classifier

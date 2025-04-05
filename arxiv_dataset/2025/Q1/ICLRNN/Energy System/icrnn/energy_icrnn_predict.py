# libraries for neural network
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Activation, Dropout, Add, LSTM, GRU, RNN, Layer
from keras import backend as K
from keras.optimizers import Adam,SGD
import tensorflow as tf
from keras import Model, regularizers, activations, initializers
from keras.constraints import Constraint
import pickle

# libraries for loading and saving mat file
import h5py
import hdf5storage

# define class and functions for ICRNN
class MyRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, input_shape_custom, **kwargs):
        self.units = units
        self.input_shape_custom = input_shape_custom
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([input_shape_custom])]
        super(MyRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer='uniform',
                                                name='recurrent_kernel',
                                                constraint=tf.keras.constraints.NonNeg(),
                                                trainable=True)
        self.D1 = self.add_weight(shape=(self.units, self.units),
                                 initializer='uniform',
                                 name='D1',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.D2 = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='uniform',
                                 name='D2',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.D3 = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='uniform',
                                 name='D3',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.V = self.add_weight(shape=(self.units, self.units),
                                 initializer='uniform',
                                 name='V',
                                 constraint=tf.keras.constraints.NonNeg(),
                                 trainable=True)
        self.built = True

    def call(self, inputs, states):
        # ICRNN
        prev_h, prev_input = states
        h = K.dot(inputs, self.kernel) + K.dot(prev_h, self.recurrent_kernel) + K.dot(prev_input, self.D2)
        h = tf.nn.relu(h)
        y = K.dot(h, self.V) + K.dot(prev_h, self.D1) + K.dot(inputs, self.D3)
        y = tf.nn.relu(y)
        return y, [h, inputs]

    def get_config(self):
        config = super(MyRNNCell, self).get_config()
        config.update({"units": self.units, "input_shape_custom": self.input_shape_custom})
        return config

# load model and scalers
model = tf.keras.models.load_model('icrnn_energy.keras', custom_objects={'MyRNNCell': MyRNNCell})
scaler_X = pickle.load(open('icrnn_scaler_X', 'rb'))
scaler_y = pickle.load(open('icrnn_scaler_y', 'rb'))

# define parameters
num_step = 5
num_dims = 4

# load data
# data has to be in the shape of (5, 2)
# 5 time steps and 2 features
with h5py.File('sample.mat', 'r') as file:
    U = file['U'][:]
    U = np.array(U)

# expand input with its negation
input = np.concatenate((U, -U), axis=1)

# normalize the input
input = scaler_X.transform(input.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)

# run prediction
output = model.predict(input)

# inverse transform the predicted y to its original scale
y_predict = scaler_y.inverse_transform(output.reshape(-1,2))

# save the predicted y in mat format
matfiledata = {} # make a dictionary to store the MAT data
matfiledata[u'Y'] = y_predict.T # u prefix for variable name = unicode format
hdf5storage.write(matfiledata, '.', 'icrnn_prediction.mat', matlab_compatible=True)
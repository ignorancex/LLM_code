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

# define class and functions for LRNN
class MyLRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, eps=0.01, gamma=0.01, beta=0.8, alpha=1, **kwargs):
        self.units = units
        self.state_size = units
        self.I = tf.eye(units)
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        super(MyLRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.C = self.add_weight(shape=(self.units, self.units),
                                      initializer='random_normal',
                                      name='C',
                                      trainable=True)
        self.B = self.add_weight(shape=(self.units, self.units),
                                                initializer='random_normal',
                                                name='B',
                                                trainable=True)
        self.U = self.add_weight(shape=(input_shape[-1], self.units),
                                                initializer='random_normal',
                                                name='U',
                                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='b',
                                    trainable=True)
        self.built = True

    def call(self, inputs, states):
        prev_h = states[0]

        A = self.beta * (self.B - tf.transpose(self.B)) + (1 - self.beta) * (self.B + tf.transpose(self.B)) - self.gamma * self.I
        W = self.beta * (self.C - tf.transpose(self.C)) + (1 - self.beta) * (self.C + tf.transpose(self.C)) - self.gamma * self.I

        h = prev_h + self.eps * self.alpha * K.dot(prev_h, A) + self.eps * tf.nn.tanh(K.dot(prev_h, W) + K.dot(inputs, self.U) + self.b)
        return h, [h]

    def get_config(self):
        config = super(MyLRNNCell, self).get_config()
        config.update({"units": self.units, "eps":self.eps, "gamma":self.gamma, "beta":self.beta, "alpha":self.alpha})
        return config

# load model and scalers
model = tf.keras.models.load_model('lrnn_energy.keras', custom_objects={'MyLRNNCell': MyLRNNCell})
scaler_X = pickle.load(open('lrnn_scaler_X', 'rb'))
scaler_y = pickle.load(open('lrnn_scaler_y', 'rb'))

# define parameters
num_step = 5
num_dims = 2

# load data
# data has to be in the shape of (5, 2)
# 5 time steps and 2 features
with h5py.File('sample.mat', 'r') as file:
    U = file['U'][:]
    U = np.array(U)

input = U

# normalize the input
input = scaler_X.transform(input.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)

# run prediction
output = model.predict(input)

# inverse transform the predicted y to its original scale
y_predict = scaler_y.inverse_transform(output.reshape(-1,2))

# save the predicted y in mat format
matfiledata = {} # make a dictionary to store the MAT data
matfiledata[u'Y'] = y_predict.T # u prefix for variable name = unicode format
hdf5storage.write(matfiledata, '.', 'lrnn_prediction.mat', matlab_compatible=True)
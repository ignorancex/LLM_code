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

# load model and scalers
model = tf.keras.models.load_model('rnn_energy.keras')
scaler_X = pickle.load(open('rnn_scaler_X', 'rb'))
scaler_y = pickle.load(open('rnn_scaler_y', 'rb'))

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
hdf5storage.write(matfiledata, '.', 'rnn_prediction.mat', matlab_compatible=True)
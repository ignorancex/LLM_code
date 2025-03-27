from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from time import time

import keras
# from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.utils import np_utils
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


__all__ = [
    'plot_prediction',
    'dataset_to_learning_vectors_without_test_train_split_control',
    'dataset_to_learning_vectors_without_test_train_split_state',
    'dataset_to_learning_vectors_control',
    'dataset_to_learning_vectors_state',
    'train_kneighbors_model',
    'train_direct_learning_model'
]


def plot_prediction(model, control_time_dim, control_dim, x_query=np.array([[-1.75,-1.75,-1.75]]), fig=None):
    pred = model.predict(x_query)

    if fig is None:
        fig = plt.figure()
    plt.plot(pred.reshape(control_time_dim,control_dim))

    return fig

def dataset_to_learning_vectors_without_test_train_split_control(samples_X, samples_U):
    X = samples_X[:,:,0]
    Y = samples_U[:,:,:]
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    return X, Y

def dataset_to_learning_vectors_without_test_train_split_state(samples_X, samples_U):
    X = samples_X[:,:,0]
    Y = samples_X[:,:,:]
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    return X, Y

def dataset_to_learning_vectors_control(samples_X, samples_U):
    X = samples_X[:,:,0]
    Y = samples_U[:,:,:]

    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def dataset_to_learning_vectors_state(samples_X, samples_U):
    X = samples_X[:,:,0]
    Y = samples_X[:,:,:]

    print(Y.shape)
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def train_kneighbors_model(X_train, Y_train, X_test, Y_test, k=None, debug=False):
    # Pick best k
    if k is None:
        regressor_score = []
        for k in range(1,10,1):
            knn_regressor_model = neighbors.KNeighborsRegressor(k)
            a = knn_regressor_model.fit(X_train, Y_train)
            score = knn_regressor_model.score(X_test, Y_test)
            regressor_score.append(score)
            debug and print("KNN Regressor (k=" + str(k) + ") - score:", knn_regressor_model.score(X_test, Y_test))

        # k = regressor_score.index(min([abs(i) for i in regressor_score])) + 1
        k = regressor_score.index(max(regressor_score)) + 1
        debug and print("Best k=", k)

    knn_regressor_model = neighbors.KNeighborsRegressor(k)
    knn_regressor_model.fit(X_train, Y_train)
    print("KNN Regressor (k=" + str(k) + ") - score:", knn_regressor_model.score(X_test, Y_test))

    return knn_regressor_model


def train_direct_learning_model(X_train, Y_train, X_test, Y_test,
                                n_hidden_layer_neurons=200,
                                n_hidden_layers=1,
                                hidden_layer_activation_function='relu',
                                batch_size=64,
                                epochs=3000,
                                debug=False):
    direct_learning_model = keras.Sequential()
    direct_learning_model.add(Dense(n_hidden_layer_neurons, input_dim=X_train.shape[1], activation=hidden_layer_activation_function))
    for _ in range(n_hidden_layers):
        direct_learning_model.add(Dense(n_hidden_layer_neurons, activation=hidden_layer_activation_function))
    direct_learning_model.add(Dense(units=Y_train.shape[1]))
    direct_learning_model.compile(loss='mse', optimizer='adam')
    direct_learning_model.summary()

    history = direct_learning_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    if debug:
        plt.plot(history.history['loss'])
        plt.tight_layout()
        plt.yscale('log')
        plt.ylim(0, history.history['loss'][0])
        plt.xlim(0, len(history.history['loss']))
        plt.show()
    
    return direct_learning_model, history

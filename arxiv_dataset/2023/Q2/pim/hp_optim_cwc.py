import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np
import pandas as pd
import tensorflow as tf 
import random
import argparse
import json
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.metrics import Metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pim import PredictionInterval

import keras_tuner as kt
from keras_tuner import Objective

datasets = {
    'yacht': 'yacht',
    'boston': 'bostonHousing',
    'energy': 'energy',
    'concrete': 'concrete',
    'wine': 'wine-quality-red',
    'kin8nm': 'kin8nm',
    'power-plant': 'power-plant',
    'naval': 'naval-propulsion-plant',
    'protein': 'protein-tertiary-structure',
    'song-year': 'YearPredictionMSD'
}

class CWC(Metric):
    def __init__(self, name='CWC', **kwargs):
        super(CWC, self).__init__(name=name, **kwargs)
        self.p_alpha = 0.95
        self.umodel = PredictionInterval(self.p_alpha)
        self.umodel.compile(loss=self.umodel.pim.loss, optimizer=Adam(learning_rate=0.005))
        self.es_pim = EarlyStopping(monitor='loss', mode='min', patience=100)
    
    def update_state(self, y_true, y_pred):
        err_va = y_pred - y_true
        rad = 0.01
        n_epochs = 100
        learning_rate = 0.005
        BETA = 1e3
        for _ in range(n_epochs):
            sigm = K.sigmoid(BETA*(rad - K.abs(err_va)))
            picp = K.mean(sigm, 0)
            grad_loss = 2*BETA* (picp - self.p_alpha) * K.mean(sigm*(1-sigm),0)
            rad = rad - learning_rate * grad_loss
        yrange = tf.reduce_max(y_true) - tf.reduce_min(y_true)
        nmpiw = 2*rad / yrange
        gamma = tf.cast(picp < self.p_alpha, tf.float32)
        mse = K.mean(err_va**2)
        self.mse_cwc = mse + nmpiw * (1 + gamma*tf.exp(-0.1*(picp - self.p_alpha)))
    
    def result(self):
        return self.mse_cwc

def load(dataset):
    path = f"data/regression/{datasets[dataset]}.txt"
    if dataset == 'song-year':
        data = pd.read_csv(path, header=None)
        x, y = data.iloc[:,1:].values, data.iloc[:,0].values.reshape(-1, 1)
    else:
        data = np.loadtxt(path)
        x, y = data[:,:-1], data[:,-1].reshape(-1, 1)
    return x, y

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def hp_model(hp, n_hidden):
    # Hyperparameters
    lr_ini = hp.Choice('lr_ini', [0.2, 0.1, 0.05, 0.02, 0.01], default=0.02)
    decay_rate = hp.Choice('decay_rate', [0.8, 0.85, 0.9, 0.95, 0.99], default=0.95)
    sigma = hp.Float('sigma', min_value=1e-5, max_value=1.0, step=0.2)
    weight_decay = hp.Choice('weight_decay', [0., 1e-3, 1e-2, 1e-1, 1.], default=1e-2)
    # Learning rate scheduler
    lr_schedule = ExponentialDecay(initial_learning_rate=lr_ini,
                                   decay_steps=50,
                                   decay_rate=decay_rate)
    # Build model
    model = Sequential()
    model.add(Dense(n_hidden, 
                    activation='relu', 
                    kernel_initializer=RandomNormal(stddev = sigma)))
    model.add(Dense(1, 
                    activation='linear',
                    kernel_initializer=RandomNormal(stddev = sigma)))
    # Compile model
    optimizer = Adam(learning_rate = lr_schedule, weight_decay = weight_decay)
    model.compile(optimizer=optimizer, loss='mse', metrics=[CWC()])
    return model

def best_model(hp_best, n_hidden):
    # Learning rate scheduler
    lr_schedule = ExponentialDecay(initial_learning_rate=hp_best['lr_ini'],
                                   decay_steps=50,
                                   decay_rate=hp_best['decay_rate'])
    # Build model
    model = Sequential()
    model.add(Dense(n_hidden, 
                    activation='relu',
                    kernel_initializer=RandomNormal(stddev = hp_best['sigma'])))
    model.add(Dense(1, 
                    activation='linear',
                    kernel_initializer=RandomNormal(stddev = hp_best['sigma'])))
    # Compile model
    optimizer = Adam(learning_rate = lr_schedule, weight_decay = hp_best['weight_decay'])
    model.compile(optimizer=optimizer, loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='boston', help='dataset to use')
    parser.add_argument('--project', type=str, default='models_cwc', help='name of the experiment')
    parser.add_argument('--seed', type=int, default=3, help='seed for random number generators')
    parser.add_argument('--n_epochs', type=int, default=2000, help='maximum number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size used for training')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity of the hyperparameter search, in [0,1,2]')
    args = parser.parse_args()
    reset_seeds(args.seed)
    # Load data
    x_al, y_al = load(args.dataset)
    # Randomly choose train and test set
    x_tr, x_te, y_tr, y_te = train_test_split(x_al, y_al, 
                            test_size=0.1, random_state=args.seed)
                            # test_size=0.1, random_state=args.seed)
    # Randomly choose validation set from 20% of previous training set
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, 
                            test_size=0.2, random_state=args.seed)
    # Standardize the data
    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)
    x_tr = s_tr_x.transform(x_tr)
    x_va = s_tr_x.transform(x_va)
    x_te = s_tr_x.transform(x_te)
    y_tr = s_tr_y.transform(y_tr)
    y_va = s_tr_y.transform(y_va)
    y_te = s_tr_y.transform(y_te)
    # Number of hidden units per dataset and batch size
    n_hidden = 100 if args.dataset in ['protein', 'song-year'] else 50
    batch_size = args.batch_size if args.dataset != 'song-year' else 1000
    # Optimize hyperparameters
    tuner = kt.Hyperband(
                lambda hp: hp_model(hp, n_hidden),
                objective = Objective('val_CWC', direction='min'),
                max_epochs = args.n_epochs,
                seed = args.seed,
                directory = args.project,
                project_name = args.dataset
            )
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    _ = tuner.search(x_tr, y_tr, 
            epochs = args.n_epochs, 
            batch_size = args.batch_size,
            validation_data = (x_va, y_va),
            callbacks = [es],
            verbose = args.verbose
        )
    print('Summary of results:')
    tuner.results_summary()
    best_hp = tuner.get_best_hyperparameters()[0].values
    with open(f'{args.project}/{args.dataset}/best_hp.json', 'w') as jsonfile:
        json.dump(best_hp, jsonfile, indent=4)
    
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np
import pandas as pd
import tensorflow as tf 
import random
import argparse
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

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
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3)
    decay = hp.Choice('decay', [0., 1e-3, 1e-2, 1e-1, 1.], default=1e-3)
    sigma = hp.Float('sigma', min_value=1e-3, max_value=1.0, step=0.2)
    dropout_rate = hp.Choice('dropout_rate', [0., 0.1, 0.25, 0.5, 0.75], default=0.)
    # Build model
    model = Sequential()
    model.add(Dense(n_hidden, 
                    activation='relu', 
                    kernel_initializer=RandomNormal(stddev = sigma)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, 
                    activation='linear',
                    kernel_initializer=RandomNormal(stddev = sigma)))
    # Compile model
    optimizer = Adam(learning_rate = learning_rate, weight_decay = decay)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

def best_model(hp_best, n_hidden):
    # Build model
    model = Sequential()
    model.add(Dense(n_hidden, 
                    activation='relu',
                    kernel_initializer=RandomNormal(stddev = hp_best['sigma'])))
    model.add(Dropout(hp_best['dropout_rate']))
    model.add(Dense(1, 
                    activation='linear',
                    kernel_initializer=RandomNormal(stddev = hp_best['sigma'])))
    # Compile model
    optimizer = Adam(learning_rate = hp_best['learning_rate'], weight_decay = hp_best['decay'])
    model.compile(optimizer=optimizer, loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='boston', help='dataset to use')
    parser.add_argument('--project', type=str, default='models', help='name of the experiment')
    parser.add_argument('--seed', type=int, default=3, help='seed for random number generators')
    parser.add_argument('--n_epochs', type=int, default=2000, help='maximum number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used for training')
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
    # Number of hidden units per dataset
    n_hidden = 100 if args.dataset in ['protein, song-year'] else 50
    # Optimize hyperparameters
    tuner = kt.Hyperband(
                lambda hp: hp_model(hp, n_hidden),
                objective = 'val_mse',
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
    
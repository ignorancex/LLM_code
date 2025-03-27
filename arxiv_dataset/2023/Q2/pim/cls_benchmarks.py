import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Layer, Input, Dense 
from tensorflow.keras.models import Model 
from tensorflow.keras.initializers import constant
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import scipy.stats as st
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from hp_optim import reset_seeds
from pim import ConfidenceInterval
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

class ModelBenchmarks(Model):

    def __init__(self, units):
        super(ModelBenchmarks, self).__init__()
        self.hidden = Dense(units, activation='tanh')
        self.pred = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.hidden(inputs)
        return self.pred(x)

def normalize_std(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    return (X - mean) / (std + 1e-7)

def normalize_minmax(X):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(X)

def run_bootstrap(features, target, 
                  train_test_ids, 
                  num_bootstrap, 
                  alpha=0.05,
                  num_epochs=1000,
                  batch_size=64):
    train_ids, test_ids = train_test_ids
    Xtrain, ytrain = features[train_ids], target[train_ids]
    Xtest, ytest = features[test_ids], target[test_ids]
    # Normalize train and test sets
    Xtrain = normalize_std(Xtrain)
    Xtest = normalize_std(Xtest)
    # Bag the predictions
    acc = []
    start_time = datetime.datetime.now()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
    for j in range(num_bootstrap):
        print(f' Training bootstrap iteration {j+1}/{num_bootstrap}')
        train_bs = resample(Xtrain)
        # Baseline model
        model = ModelBenchmarks(Xtrain.shape[1])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_bs, ytrain, validation_data=(Xtest, ytest),
                  epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[es])
        acc.append(model.evaluate(Xtest, ytest, verbose=0)[1])
    # Confidence interval for ACC from order statistics
    p = (0.5 * alpha) * 100
    lower = max(0.0, np.percentile(acc, p))
    p = (1.0 - alpha + 0.5 * alpha) * 100
    upper = min(1.0, np.percentile(acc, p))
    end_time = datetime.datetime.now()
    time = (end_time - start_time).total_seconds()
    return np.mean(acc), upper-lower, time

def run_pim(ytrue, ypred, p=0.95, num_classes=2, threshold='fix', do_plot=False):
    umodel = ConfidenceInterval(p, num_classes, threshold)
    umodel.compile(loss=umodel.pim.loss, optimizer=optimizers.Adam(lr=0.005))
    hist = umodel.fit([ytrue, ypred], [ytrue, ypred], epochs=100, batch_size=ytrue.shape[0], 
                      callbacks=[umodel.early_stop], verbose=0)
    if do_plot:
        plt.plot(hist.history['loss'])
        plt.show()
    return umodel.pim.binary_umetrics(ypred, ytrue)

def run_single_model(features, target, 
                     train_test_ids, 
                     alpha=0.05,
                     num_epochs=1000,
                     batch_size=64):
    print(' Training single model')
    train_ids, test_ids = train_test_ids
    Xtrain, ytrain = features[train_ids], target[train_ids]
    Xtest, ytest = features[test_ids], target[test_ids]
    # Normalize train and test sets
    Xtrain = normalize_std(Xtrain)
    Xtest = normalize_std(Xtest)
    # Baseline model
    start_time = datetime.datetime.now()
    model = ModelBenchmarks(Xtrain.shape[1])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    _ = model.fit(Xtrain, ytrain, validation_data = (Xtest, ytest),
                  epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    ytrue = np.expand_dims(ytest,-1)
    ypred = model.predict(Xtest)
    # Run PIM to estimate confidence intervals
    umetrics = run_pim(ytrue, ypred)
    end_time = datetime.datetime.now()
    time = (end_time - start_time).total_seconds()
    return ytrue, ypred, umetrics, time   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sonar')
    parser.add_argument('--data_dir', type=str, default='data/classification/processed/')
    parser.add_argument('--n_ens', type=int, default=30, help='number of experiments')
    parser.add_argument('--n_bs', type=int, default=20, help='number of bootstrap iterations')
    args = parser.parse_args()
    sss = StratifiedShuffleSplit(n_splits=args.n_ens, train_size=0.8, random_state=0)
    # Load dataset and separate features from targets
    data = pd.read_csv(args.data_dir + f'{args.dataset}.csv', header=None)
    features, target = data.values[:,:-1], data.values[:,-1]
    # Run models in ensemble members
    results_bs = []
    results_pim = []
    ytrue_single_model = []
    ypred_single_model = []
    for n, train_test_ids in enumerate(sss.split(features, target)):
        reset_seeds(n)
        print(f'Running experiment {n+1}/{args.n_ens}')
        ytrue, ypred, umetrics, time_sm = run_single_model(features, target, train_test_ids)
        acc, uacc, time_bs = run_bootstrap(features, target, train_test_ids, args.n_bs)
        ytrue_single_model.append(np.squeeze(ytrue.T))
        ypred_single_model.append(np.squeeze(ypred.T))
        results_pim.append(umetrics['ACC']+[time_sm])
        results_bs.append([acc, uacc, time_bs])
    # Save results
    results_dir = Path(args.data_dir).parent / 'results'
    bs_res_path = results_dir / f'{args.dataset}_bs.csv'
    sm_res_path = results_dir / f'{args.dataset}_sm.npy'
    pim_res_path = results_dir / f'{args.dataset}_pim.csv'
    np.save(sm_res_path, [ytrue_single_model, ypred_single_model])
    pd.DataFrame(results_pim, columns=['acc', 'uacc', 'time']).to_csv(pim_res_path, index=False)
    pd.DataFrame(results_bs, columns=['acc', 'uacc', 'time']).to_csv(bs_res_path, index=False)
        
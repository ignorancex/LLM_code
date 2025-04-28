from multiprocessing import Pool, Lock
import os
import sys

from datetime import datetime

import sys
import importlib
sys.path.insert(0, '../')

import mortalityForecast as mf

import numpy as np
import pandas as pd
from itertools import product
from itertools import cycle
import psutil 

first_year = 1933
train_length = 60
forecast_length = 10
last_year = 2020

param_tuple = tuple()

start_years = range(first_year, last_year - train_length - forecast_length + 2)
countries = [mf.data_loader.USA]
nn_layers = [1]
latent_dims = [3,4,5,6,7]

for start_year, country, nn_layer, latent_dim in product(start_years, countries, nn_layers, latent_dims):
    param_tuple = param_tuple + ({
    'country' : country,
    'first_year_train' : start_year,
    'last_year_train' : start_year + train_length - 1,
    'first_year_test' : start_year + train_length,
    'last_year_test' : start_year + train_length + forecast_length-1,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : nn_layer,
    'latent_dim' : latent_dim},)

for idx, param in enumerate(param_tuple):
    param['id'] = idx +1    

l = Lock()

def train(param):

    exposure_all, deaths_all = mf.data_loader.load_data(param, train = True, test = True)
    exposure_train, deaths_train = mf.data_loader.load_data(param, train = True, test = False)
    exposure_test, deaths_test = mf.data_loader.load_data(param, train = False, test = True)

    model = mf.Mortality(param)

    file_name = "../trainedModels/%s_%d_%d_%s_%d_%d_%d" %(param['country'].name, param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['nn_layers'], param['latent_dim'])

    model = model.fit(exposure = exposure_train, 
                        deaths = deaths_train, 
                        num_steps = 20000,# 60000, 
                        log_freq = 1000, 
                        lr = 0.01,
                        checkpoint_freq = 1000,
                        lr_decay = 0.9999,
                        save_file = file_name)

    log_score = model.log_score(exposure_test, deaths_test, mc_samples = 10000)

    data = list(
    zip(
        cycle(['sweden']),
        cycle([param['first_year_train']]),
        cycle([param['last_year_train']]),
        cycle([param['sex']]),
        cycle([param['max_age']]),
        cycle([param['nn_layers']]),
        cycle([param['latent_dim']]),
        range(1,len(log_score)+1),
        log_score,
        cycle([datetime.now().strftime("%d/%m/%Y")]),
        cycle([datetime.now().strftime("%H:%M:%S")])
        )
    )
    
    df = pd.DataFrame(data,
                  columns = ['Country', 'first_year_train', 'last_year_train', 'Sex','max_age', 'nn_layers', 'latent_dim', 'Forecast horizon', 'Log score', 'Date', 'Time'])


    out_path = '../trainedModels/results/results-230310' + '.csv'
    
    l.acquire()
    try:
        df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index = False)
        print(param['id'])
    finally:
        l.release()

if __name__ == '__main__':

    n_cores = psutil.cpu_count(logical = True)
    print(len(param_tuple))
    print(n_cores)


    with Pool(n_cores) as p:
        p.map(train, param_tuple, chunksize=1)
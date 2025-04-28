from multiprocessing import Pool, Lock
import os
import sys

from datetime import datetime

import sys
import importlib
sys.path.insert(0, '../')

import mortalityForecast as mf

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from itertools import product
from itertools import cycle
import psutil 

first_year = 1920
train_length = 60
forecast_length = 10
last_year = 2021

param_tuple = tuple()

start_years = range(first_year, last_year - train_length - forecast_length + 2)
countries = [mf.data_loader.SWEDEN]
nn_layers = [1]
latent_dims = [1]

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

    exposure_all, deaths_all = mf.load_data(param, train = True, test = True)
    exposure_train, deaths_train = mf.load_data(param, train = True, test = False)
    exposure_test, deaths_test = mf.load_data(param, train = False, test = True)

    model_string = "%s_%d_%d_%s_%d_%d_%d" % (param['country'].name, param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['nn_layers'], param['latent_dim'])
    file_name = "../trainedModels/" + model_string


    model = mf.Mortality(param)
    model = model.load(file_name = file_name, param = param, exposure = exposure_train, deaths = deaths_train )

    log_score = model.log_score_naive(exposure_test, deaths_test)

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
        cycle([datetime.now().strftime("%H:%M:%S")]),
        cycle(['naive'])
        )
    )
    
    df = pd.DataFrame(data,
                  columns = ['Country', 'first_year_train', 'last_year_train', 'Sex','max_age', 'nn_layers', 'latent_dim', 'Forecast horizon', 'Log score', 'Date', 'Time', 'Model'])


    out_path = 'results_naive_saturated' + '.csv'
    
    l.acquire()
    try:
        df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index = False)
        print(param['id'])
    finally:
        l.release()

if __name__ == '__main__':

    n_cores = psutil.cpu_count(logical = False)

    with Pool(n_cores) as p:
        p.map(train, param_tuple, chunksize=1)
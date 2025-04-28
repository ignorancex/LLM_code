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

first_year = 1931
train_length = 60
forecast_length = 10
last_year = 2021

param_tuple = tuple()

start_years = range(first_year, last_year - train_length - forecast_length + 2)
countries = [mf.data_loader.SWEDEN]

num_bases = [20]
taus= [10.0]
latent_dims = [3,5]

for start_year, country, latent_dim, tau, num_basis in product(start_years, countries, latent_dims, taus, num_bases):
    param_tuple = param_tuple + ({
    'country' : country,
    'first_year_train' : start_year,
    'last_year_train' : start_year + train_length - 1,
    'first_year_test' : start_year + train_length,
    'last_year_test' : start_year + train_length + forecast_length-1,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'model' : 'radial_basis',
    'num_basis' : num_basis,
    'tau' : tau,
    'latent_dim' : latent_dim},)

for idx, param in enumerate(param_tuple):
    param['id'] = idx +1    

l = Lock()

def train(param):

    exposure_all, deaths_all = mf.data_loader.load_data(param, train = True, test = True)
    exposure_train, deaths_train = mf.data_loader.load_data(param, train = True, test = False)
    exposure_test, deaths_test = mf.data_loader.load_data(param, train = False, test = True)

    model = mf.Mortality(param)


    model_string = "%s_%s_%d_%d_%s_%d_%d_%d" %(
        param['country'].name, 
        'radial_basis', 
        param['first_year_train'], 
        param['last_year_train'], 
        param['sex'], 
        param['max_age'], 
        param['num_basis'], 
        param['latent_dim']
        )
    file_name = "../trainedModelsRadial/" + model_string

    model = model.fit(exposure = exposure_train, 
                        deaths = deaths_train, 
                        num_steps = 5000,# 60000, 
                        log_freq = 1000, 
                        lr = 0.01,
                        checkpoint_freq = 1000,
                        lr_decay = 1.0,
                        save_file = file_name)

    log_score = model.log_score(exposure_test, deaths_test, mc_samples = 10000)

    data = list(
    zip(
        cycle(['sweden']),
        cycle([param['first_year_train']]),
        cycle([param['last_year_train']]),
        cycle([param['sex']]),
        cycle([param['max_age']]),
        cycle([param['num_basis']]),
        cycle([param['tau']]),
        cycle([param['latent_dim']]),
        range(1,len(log_score)+1),
        log_score,
        cycle([datetime.now().strftime("%d/%m/%Y")]),
        cycle([datetime.now().strftime("%H:%M:%S")]),
        cycle(['radial_basis']),
        )
    )
    
    df = pd.DataFrame(data,
                  columns = ['country', 'first_year_train', 'last_year_train', 'sex','max_age', 'num_basis','tau', 'latent_dim', 'forecast_horizon', 'log_score', 'date', 'time','model'])


    out_path = '../trainedModelsRadial/results/results_radial' + '.csv'
    
    l.acquire()
    try:
        df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index = False)
        print(param['id'])
    finally:
        l.release()

if __name__ == '__main__':

    n_cores = 3 #psutil.cpu_count(logical = True) -1
    print(len(param_tuple))
    print(n_cores)


    with Pool(n_cores) as p:
        p.map(train, param_tuple, chunksize=1)
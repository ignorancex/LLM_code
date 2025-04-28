# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:36:21 2022

@author: David J. Kedziora
"""

import os
import pandas as pd
import numpy as np

from time import time

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

import calc
import plot

filename_prefixes = [
    # "1p2uW_3000cps_time bin width 128 ps",
    # "2p5uW_4000cps",
    # "4uW_4100cps",
    # "8uW_5100cps",
    # "10uW_6000cps",
    # "10uW_12000cps",
    # "20uW_7000cps",
    "30uW_7000cps"
    ]
folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"
random_seed = 0

# The number of traces generated per datafile.
# Each trace shows how g2(0) estimates change with greater sampling.
num_traces = 100

for filename_prefix in filename_prefixes:
    
    # Prepares save destination for any plots and storable results.
    plot_prefix = folder_plots + filename_prefix
    save_prefix = folder_saves + filename_prefix + "_traces_" + str(num_traces) + "_seed_" + str(random_seed)
    
    stat_labels = ["avg", "std", "min", "max"]
    df_trace_g2zero = {}
    df_trace_amp = {}
    df_trace_bg = {}
    try:
        for stat_label in stat_labels:
            df_trace_g2zero[stat_label] = pd.read_pickle(save_prefix + "_g2zero_" + stat_label + ".pkl")
            df_trace_amp[stat_label] = pd.read_pickle(save_prefix + "_amp_" + stat_label + ".pkl")
            df_trace_bg[stat_label] = pd.read_pickle(save_prefix + "_bg_" + stat_label + ".pkl")
        df_trace_amp["mpe"] = pd.read_pickle(save_prefix + "_amp_mpe.pkl")
        
        print("%i previously saved traces for seed %i loaded." 
          % (num_traces, random_seed))
    except:
        print("Cannot find %i traces for seed %i." % (num_traces, random_seed))
        raise

g2zero = np.nan_to_num(np.ravel(np.array(df_trace_g2zero["avg"].iloc[0,:])),
                       nan=2.0, posinf=2.0, neginf=2.0)
data = {
    "g2zero": g2zero
}

# Set task type and forecast length.
forecast_length = int(g2zero.shape[0]/4)
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length = forecast_length))

# init model for the time-series forecasting
model = Fedot(problem="ts_forecasting", task_params=task.task_params,
              timeout=10)

# run AutoML model design
pipeline = model.fit(features = data, target = g2zero)

# use model to obtain in-sample forecast
model.predict(data)


# get metrics
metric = model.get_metrics(g2zero[-forecast_length:])

pipeline.show()
model.plot_prediction(target="g2zero")
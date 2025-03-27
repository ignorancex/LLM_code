#!/usr/bin/env python3
# coding: utf-8

"""
This work is licensed under CC BY 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/

Created on Mon July 15 2024 (version 2). Updated August 2024, November 1 (v3),
and Nov 8 (v4)

Forecasting mobile cellular traffic load

A light-weight LSTM model for predicting the number of call arrivals at a BS
serving a sector of a highway.
The machine learning model is fed with (multivariate) data comprised by 
(road traffic and) mobile cellular network metrics. 
The flow and speed are measured in the vicinity yet outside the range of the BS. 
At a given time slot, the measurements are from outside (road variables) and 
inside (mobile network variables) the BS. 
The flow and speed (road variables) measured at time slot 't', 
are observed at the BS at time slot 't+1'. 
In other words, they impact the mobile cellular variables from 't+1' on.


@author: vesseln1
"""



__title__			 = 'Mobile traffic forecasting'
__description__		 = 'LSTM model for short-term trafic load predictions.'
__version__			 = '4.0.0'
__date__			 = 'July 2024'
__author__			 = 'Natalia Vesselinova'
__author_email__	 = 'natalia.vesselinova@aalto.fi'
__institution__ 	 = 'Alto University'
__department__		 = 'Mathematics and Systems Analysis'
__url__				 = 'https://version.aalto.fi/gitlab/vesseln1/forecasting/'
__license__          = 'CC BY 4.0'


import os
import datetime
import argparse
from os.path import join
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

import ruamel.yaml
yaml = ruamel.yaml.YAML()

print("\n  ̵Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', '-dir', required=True, type=str,
                    help='Path to the directory with the data.')
parser.add_argument('--outputRoot', '-o', required=True, type=str,
                    help='Path to the output directory.')
parser.add_argument('--dataFile', '-d',  required=False, type=str,
                    help='The file with input data that feeds the ML model.', default='pems.csv')
parser.add_argument('--runs', '-r', type=int,
                    help='Number of runs per dataset', default=10)
parser.add_argument('--trainEpochs', '-e', type=int,
                    help='Number of training epochs', default=100)
parser.add_argument('--earlyStoppingPatience', '-s',  type=int, 
                    help='Patience for the early stopping. Defaults to the number of training epochs', default=100)
# 24 hours x 5 minute granularity * 5 workdays (weekends excluded)
parser.add_argument('--dataPointsW', '-p', type=int,
                    help='Number of data points per week', default=1441)
parser.add_argument('--numTrainW', '-tr', type=int,
                    help='Number of training weeks', default=12)
parser.add_argument('--numValW',  '-val', type=int,
                    help='Number of validation weeks', default=6)
parser.add_argument('--numTestW', '-te', type=int,
                    help='Number of test weeks', default=6)
parser.add_argument('--samplingRate', type=int,
                    help='Every sample (=1), every other sample (=2), etc.?', default=1)
parser.add_argument('--learningRate', '-lr', type=float,
                    help='Learning rate of the RMSOptimiser.', default=0.001)
parser.add_argument('--sequenceLength', '-w', type=int,
                    help='Look-back / historical window / number of consequtive samples on which the prediction is based', default=6)
parser.add_argument('--future', type=int,
                    help='How far in the future are we predicting (one timeslot, more)?', default=1)
parser.add_argument('--batchSize', '-b', type=int, default=6)
parser.add_argument('--numLSTMUnits', '-u', type=int, default=16)
parser.add_argument('--features', '-f',  required=False, type=str, help='The features to be used.', nargs='+', default=['Flow (Veh/5 Minutes)', 'Speed Level', 'Total New Calls', 'Total HO Calls', 'Total Number Calls'] )
parser.add_argument('--week1', '-w1', required=False, type=int, help='The first week of data. Used when not all of the data in the dataFile is to be employed.')
parser.add_argument('--week2', '-w2', required=False, type=int, help='The last week of data. Used when not all of the data in the dataFile is to be employed.')


args = parser.parse_args()

# Arguments for the input and output directories and the file with the data
data_root    = args.dataRoot
output_root  = args.outputRoot
data_file    = args.dataFile

# Arguments specific to the simulation runs
runs         = args.runs
train_epochs = args.trainEpochs
patience     = args.earlyStoppingPatience
learn_rate   = args.learningRate

# Arguments specific to the data
data_w       = args.dataPointsW
begin_w      = args.week1
end_w        = args.week2
features     = args.features        # a list of the features to be used
num_features = len(features)

# Arguments specific to the ML process -- training, testing, validation
num_lstm     = args.numLSTMUnits
num_train_w  = args.numTrainW
num_val_w    = args.numValW
num_test_w   = args.numTestW

# Arguments specific to the prediction: 
# sampling, historical window, future window, batch and shuffling
sample_rate= args.samplingRate
seq_len      = args.sequenceLength
future       = args.future
batchSize    = args.batchSize


# A dictionary with the arguments values
args_dict = vars(args)           


# Define the slicing parameters for the Keras timeseries_dataset_from_array
# the first target position
delay = sample_rate * (seq_len + future - 1) 
    

# A function that standardises the input data and takes care when the variance
# of a variable is zero by adding a small constant to it.
def standardize_data(data, mean, stddev, epsilon=1e-7):
    '''
        Standardises the data set 

        Input parameters:
        data    : numpy array, the data to be standardised (the pems data)
        
        mean    : the mean to use (that of the training data)
        
        stddev  : the standard deviation (of the training data)
    
        epsilon : a small constant added whenever a variable has a 0 variance.
                    It is optional. The default is 1e-7.

        Returns the standardised data. 
    '''
    stddev[stddev == 0] = epsilon
    data -= mean
    data /= stddev
    
    return data


# 'time_series_dataset_from_array': Creates a dataset of sliding windows over
# a timeseries provided as an **array** (or eager tensor)
# A wrapper function
def timeseries_batch(historical_data, target_data, shuffling=False
                    ):
    """ 
        Since all the data are processed using the same
        input parameters: sequence_length, shuffle and batch_size, 
        this is a wrapper function based on the Tensor Flow timeseries function
    """
    return keras.preprocessing.timeseries_dataset_from_array(
        data=historical_data,
        targets=target_data,
        sequence_length=seq_len,
        shuffle=shuffling,
        sampling_rate=sample_rate,
        # seed=int((datetime.datetime.now())timestamp()),s
        batch_size=batchSize)



# A wrapper function
def time_series(weeks, historical_data, shuffling=False
               ):
    """ 
       To easily loop through all the weeks (>=1).
    """
    concatenated_dataset = None   
    for w in range(weeks):
        begin = w * data_w
        end = begin + data_w
        week = historical_data[begin:end]
        dataset = timeseries_batch(week[:-future], week[delay:, -1], shuffling)
                
        if concatenated_dataset is None:
            concatenated_dataset = dataset
        else:
            concatenated_dataset = concatenated_dataset.concatenate(dataset)
   
    return concatenated_dataset


# A data range selection function
def data_selection(begin_week, end_week, data_set
        ):
    """
    Selects the data from the data_file that corresponds to the period 
    from begin_w to end_w. Checks that the selected period measured in
    number of weeks is at most as large as the provided data_set

    Input arameters
    
    begin_w     : int; the start week of the chosen period.
    end_w       : int; the end week of the chosen period.
    data_set    : numpy array; data set from which data from begin_w to
                    end_w is selected

    Returns a pandas dataframe with the selected period
    """
    num_week = int(data_set.shape[0] / data_w)

    # Ensure there are enough weeks in the dataset 
    if end_week > num_week:   
        raise ValueError(f"The dataset in {data_file} consists of {num_w} weeks -- {end_week} is beyond the range." )
               
    return data_set[(begin_week - 1) * data_w : end_week * data_w]




start = datetime.datetime.now()
print(start)


# To make the GPU ops as deterministic as possible
tf.config.experimental.enable_op_determinism()



# Change the directory to the folder with the data
os.chdir(data_root)

# Reading all the data at once into a pandas dataframe
df_pems = pd.read_csv(data_file, header=0, usecols=features)
args_dict['number of weeks in the dataset (input file)'] = int(df_pems.shape[0] / data_w)

# Converting the pandas dataframe into a numpy array
pems = df_pems.to_numpy(dtype=float)

# Check if only a fraction of the data will be used
if begin_w is not None and end_w is not None:
    pems = data_selection(begin_w, end_w, pems)
    
num_w = int(pems.shape[0] / data_w)
if num_train_w + num_val_w + num_test_w > num_w:
    raise ValueError(f"The dataset in {data_file} consists of {num_w} weeks -- " + 
                     f"{num_train_w + num_val_w + num_test_w} is beyond the range." )
    

# Saving the input values into an yaml file.
args_dict['number of weeks employed in machine learning'] = num_w
with open('input_ml.yml', 'w') as f:
    yaml.dump(args_dict, f)


# Find the mean and standard deviation of the training data
train_len = data_w * num_train_w     # determine the length of the training data
train_raw = pems[:train_len]         # keep chronological order for time-seriees
train_mean = train_raw.mean(axis=0)
train_std = train_raw.std(axis=0)

# Normalising / standardising the data
pems = standardize_data(pems, train_mean, train_std)

runstart = datetime.datetime.now()

for run in range(runs):
    
    # Setting the random seed of Python, NumPy and backend
    keras.utils.set_random_seed(run)
    args_dict['seed'] = run

    print("Run: ", run)
    print(datetime.datetime.now())

    # Create a subfolder / directory inside the output_root and use
    # the run number as a name to save all results from this run inthere
    subdir = str(run)
    os.makedirs(join(output_root, subdir), exist_ok=True)


    ### Splitting the data and preparing it for training
    # Create a `Dataset` object that yields batches
    # of data from the past `seq_len` minutes along with a target
    # number of calls 'future' minutes in the future from the first week.
    train_data = time_series(num_train_w, pems[:train_len], True)
   
    # Repeating the same steps for the validation and test datasets
    # Validation dataset preparation
    val_len = data_w * num_val_w   #determine the length of the validation data
    val = pems[train_len : (train_len + val_len)]
    val_data = time_series(num_val_w, val) 
    

    # Change the directory to the results folder
    os.chdir(output_root)
    # Change to this run's subfolder
    os.chdir(subdir)
    
    # Checking if this run has already been successfully completed
    if os.path.isfile('AllIsFine.txt'):
        print(f"Run {run} was completed already." )
        continue

    

    # DL Model: LSTM
    inputs = keras.Input(shape=(seq_len, num_features)) 
    x = layers.LSTM(num_lstm)(inputs)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    

    model_name = f'{num_lstm}_lstm.keras'
    # Save only the best model from all epochs
    model_checkpoint_callback = [
        keras.callbacks.ModelCheckpoint(model_name,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        verbose=1)
    ]

    # Early stopping for finding the best model
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=patience,
                                  mode="min",
                                  verbose=1)
    
    # Configure the model
    opt = keras.optimizers.RMSprop(learning_rate=learn_rate)
    model.compile(optimizer=opt, loss="mse", metrics=[
                  "mae", "mse", "mape", keras.metrics.RootMeanSquaredError()])

    # Train the model
    history = model.fit(train_data,
                        epochs=train_epochs,
                        validation_data=val_data,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])

    model = keras.models.load_model(model_name)

    
    ### Test 
    # Test dataset preparation
    if num_test_w != 0:
        test_len = data_w * num_test_w   # determine the length of the test data
        test = pems[train_len+val_len:train_len+val_len+test_len]
        test_data = time_series(num_test_w, test)
        # Testing   
        with open(f'test_results-{run}.csv', 'w') as resultsFile:    
            resultsFile.write("MAE, MSE, MAPE, RMSE,\n")
            err = model.evaluate(test_data)
            resultsFile.write(f"{err[1]:.10f}, {err[2]:.10f}, {err[3]:.10f}, {err[4]:.10f}")
        test_data = None
    
        # Plotting the results
        loss = history.history["mae"]
        val_loss = history.history["val_mae"]
        epochs = range(1, len(loss) + 1)
        plt.figure(run)
        plt.plot(epochs, loss, "bo", label="Training MAE")
        plt.plot(epochs, val_loss, "r*", label="Validation MAE")
        plt.title("Training and validation MAE")
        plt.legend()
        plt.savefig(f'{run}.png' )
    
        with open("AllIsFine.txt", 'w') as file:
            file.write(str(datetime.datetime.now()))
            file.write("\nThe execution of the run took " + str(datetime.datetime.now() - runstart))     
            
    elif num_test_w == 0:
        with open("TrainedModelSaved.txt", 'w') as file:
            file.write(str(datetime.datetime.now()))
            file.write("\nThe execution of the run took " + str(datetime.datetime.now() - runstart))
            
    # Go back to the main directory, output_root
    os.chdir('..')
    runstart = datetime.datetime.now()
   

plt.figure(run+1)
corr_df = df_pems.corr(method='pearson')
df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(bool))
hmap = sns.heatmap(df_lt, annot=True, cmap="PuOr")
plt.savefig('corr.png')


print(datetime.datetime.now())
print("The execution of the program took: ", datetime.datetime.now() - start)
print("All runs accomplished.")

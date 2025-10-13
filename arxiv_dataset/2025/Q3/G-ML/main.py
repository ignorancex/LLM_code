"""
Created on Mon Feb 03 2025, 11:20:15

@author: Amoy Ashesh
"""
import pandas as pd
from load import mark_data
from split import split
from parameters import instruct, algorithms
import pickle

if instruct.load_data:
    # Load previously marked data
    dt = pd.read_csv('marked\marked_data.csv')
else:
    # Mark data
    dt = mark_data(save_data=True)

# Split the dataset     
X_train, X_test, y_train, y_test = split(dt)

# Training
if instruct.load_models:
    # Load pretrained models
    with open('raw_models.pkl', 'rb') as f:
        loaded_models = pickle.load(f)
    # Load the SMOTE models
    with open('re_models_smote.pkl', 'rb') as f:
        smote_loaded_models = pickle.load(f)
else:    
    # Train ML model(s)
    algs = algorithms()
    algs.train(X_train, y_train)
    algs.fill_predictions(X_test)
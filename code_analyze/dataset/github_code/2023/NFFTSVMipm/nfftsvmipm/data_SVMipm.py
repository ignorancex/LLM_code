"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

###############################################
# include data folder into the path
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data folder (one level up from the current directory)
data_dir = os.path.join(current_dir, '..', 'data')

##################################################################################

# SUSY Dataset
def susy(num=0):
    """
    Load the SUSY data set and prepare it for usage.
    
    Note
    ----
    If num!=0, this function returns a balanced data set, i.e. a data set with the same number of samples in both classes.
        
    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included and the data set is not balanced out.
        
    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The training target vector incorporating the training labels.
    y_test : ndarray
        The test target vector incorporating the test labels.
    """
    # Construct the absolute path to the specific file in the data folder
    file_path_susy = os.path.join(data_dir, 'SUSY.csv')
    # read dataset
    df = pd.read_csv(file_path_susy, header=None)
    
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    
    # convert dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()
    
    # reshape y
    y = np.reshape(y, (X.shape[0],))
    
    # convert labels: 0 -> -1 and save label for all indices
    idx_signal = []
    idx_background = []
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
            idx_background.append(i)
        else:
            idx_signal.append(i)
            
    if num != 0:
        r1 = random.sample(idx_signal, num)
        r2 = random.sample(idx_background, num)
        r_samples = r1 + r2
        
        X = X[r_samples,:]
        y = y[r_samples]
    
    # split data in train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    
    return X_train, X_test, y_train, y_test


##################################################################################
    
# HIGGS Dataset
def higgs(num=0):
    """
    Load the HIGGS data set and prepare it for usage.
    
    Note
    ----
    If num!=0, this function returns a balanced data set, i.e. a data set with the same number of samples in both classes.
        
    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included and the data set is not balanced out.
        
    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The training target vector incorporating the training labels.
    y_test : ndarray
        The test target vector incorporating the test labels.
    """
    # Construct the absolute path to the specific file in the data folder
    file_path_higgs = os.path.join(data_dir, 'HIGGS.csv')
    # read dataset
    df = pd.read_csv(file_path_higgs, header=None)
    
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    
    # convert dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()
    
    # reshape y
    y = np.reshape(y, (X.shape[0],))
    
    # convert label: 0 -> -1 and save label for all indices
    idx_higgs = []
    idx_else = []
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
            idx_else.append(i)
        else:
            idx_higgs.append(i)
             
    if num != 0:
        r1 = random.sample(idx_higgs, num)
        r2 = random.sample(idx_else, num)
        r_samples = r1 + r2
        
        X = X[r_samples,:]
        y = y[r_samples]
    
    # split data in train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    
    return X_train, X_test, y_train, y_test


##################################################################################
    
# cod-rna Dataset
def cod_rna(num=0):
    """
    Load the cod-rna data set and prepare it for usage.
    
    Note
    ----
    If num!=0, this function returns a balanced data set, i.e. a data set with the same number of samples in both classes.
        
    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included and the data set is not balanced out.
        
    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The training target vector incorporating the training labels.
    y_test : ndarray
        The test target vector incorporating the test labels.
    """
    # get the number of samples to be included per class in the training and the test set
    num_cod = int(num/2)
    
    # Construct the absolute path to the specific file in the data folder
    file_path_cod_train = os.path.join(data_dir, 'cod-rna-train.txt')
    file_path_cod_val = os.path.join(data_dir, 'cod-rna-val.txt')
    file_path_cod_test = os.path.join(data_dir, 'cod-rna-test.txt')
    
    ## read training and validation data set
    [X_t,y_t] = load_svmlight_file(file_path_cod_train)
    
    [X_v,y_v] = load_svmlight_file(file_path_cod_val)
    
    # concatenate training and validation set to training data set
    X_train = np.concatenate((X_t.todense(),X_v.todense()), axis=0)
    y_train = np.hstack((y_t,y_v))
    
    # save label for all indices
    idx_pos_train = []
    idx_neg_train = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            idx_pos_train.append(i)
        else:
            idx_neg_train.append(i)
 
    if num != 0:
        r1_train = random.sample(idx_pos_train, num_cod)
        r2_train = random.sample(idx_neg_train, num_cod)
        r_samples_train = r1_train + r2_train
        
        X_train = X_train[r_samples_train,:]
        y_train = y_train[r_samples_train]
    
    ## read test dataset
    [X_test,y_test] = load_svmlight_file(file_path_cod_test)
    
    X_test = X_test.todense()

    # save label for all indices
    idx_pos_test = []
    idx_neg_test = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            idx_pos_test.append(i)
        else:
            idx_neg_test.append(i)
   
    if num != 0:
        r1_test = random.sample(idx_pos_test, num_cod)
        r2_test = random.sample(idx_neg_test, num_cod)
        r_samples_test = r1_test + r2_test
        
        X_test = X_test[r_samples_test,:]
        y_test = y_test[r_samples_test]
    
    return X_train, X_test, y_train, y_test
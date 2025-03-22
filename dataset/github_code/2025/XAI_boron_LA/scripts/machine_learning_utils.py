import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold

from rdkit.Chem import Descriptors

from sklearn.datasets import make_classification

from sklearn.multioutput import MultiOutputRegressor


## Data preprocessing

def delete_columns(X_train, X_test):
    cols_to_delete = []
    for i in range(X_train.shape[1]):
        if X_train[:,i].std() == 0.0 :
            cols_to_delete.append(i)
    return np.delete(X_train, cols_to_delete, axis = 1), np.delete(X_test, cols_to_delete, axis = 1)

def split(X,Y,test_size, shuffle = True): #split&scale the dataset
                          #All columns should be scaled to have 0 mean and unit standard deviation
    """ X: descriptors of the molecules ; type : array or list
        Y: property of interest, e.g. FIA ; type : array or list
        returns: matrixes splitting the dataset between train and test"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, shuffle = shuffle)
    var_selector =  VarianceThreshold()
    X_train = var_selector.fit_transform(X_train)
    X_test = var_selector.transform(X_test)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train, X_test = delete_columns(X_train, X_test)
    return(X_train, X_test, Y_train, Y_test)



#functions to access Y_pred_test

def train_predict(X_train, X_test, Y_train, Y_test, model):
    var_selector =  VarianceThreshold()
    X_train = var_selector.fit_transform(X_train)
    X_test = var_selector.transform(X_test)
    scaler = StandardScaler().fit(X_train) ## descriptors are scaled
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    # it is very important to fit the preprocessing only 
    #on train data to avoid data leakage
    
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    MAE = mean_absolute_error(Y_test, model.predict(X_test))
    return(Y_pred_test,MAE)

def K_Fold_model_evaluation(model,X,Y,n_fold, n_repet):
    
    total_Y_test = []
    total_Y_pred_test = []
    list_MAE = []
    
    for i in range(n_repet):
        KF = KFold(n_splits=n_fold, shuffle=True)# do not fix the random state or it will give the same split for each n_rep
        for i, (train_index, test_index) in enumerate(KF.split(X,Y)):
            X_train = X[train_index]
            X_test = X[test_index]
            Y_train = Y[train_index]
            Y_test = Y[test_index]

            Y_pred_test,MAE = train_predict(X_train, X_test, Y_train, Y_test, model)

            for elt in Y_test :
                total_Y_test.append(elt)
            for elt in Y_pred_test :
                total_Y_pred_test.append(elt)
            
            list_MAE.append(MAE)    
    
    return(total_Y_test, total_Y_pred_test, list_MAE)


def evaluate_model(model, X,Y, n_rep =10):
    pipeline = Pipeline(steps=[('selector', VarianceThreshold()),('scaler', StandardScaler()),('m',model)])
    cv = RepeatedKFold(n_splits=10, n_repeats=n_rep)
    scores = cross_val_score(pipeline, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return(scores)

def evaluate_model_fp(model, X,Y, n_rep =10): #no scaling for fp because bit-vectors
    pipeline = Pipeline(steps=[('selector', VarianceThreshold()), ('m',model)])
    cv = RepeatedKFold(n_splits=10, n_repeats=n_rep)
    scores = cross_val_score(pipeline, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return(scores)

def evaluate_model_test_set(model, X_train, Y_train, X_test, Y_test):
    selector = VarianceThreshold()
    selector.fit(X_train) ## features that have same value 
    X_train = selector.transform(X_train) # for every molecule are eliminated
    X_test = selector.transform(X_test) 
    scaler = StandardScaler().fit(X_train) ## descriptors are scaled
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    # it is very important to fit the preprocessing only 
    #on train data to avoid data leakage
    
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    MAE = mean_absolute_error(Y_test, model.predict(X_test))
    
    return(MAE)

def evaluate_model_test_set_fp(model, X_train, Y_train, X_test, Y_test):
   
    selector = VarianceThreshold()
    selector.fit(X_train) ## features that have same value 
    X_train = selector.transform(X_train) # for every molecule are eliminated
    X_test = selector.transform(X_test) # it is very important to fit the preprocessing only 
    #on train data to avoid data leakage
    
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    MAE = mean_absolute_error(Y_test, model.predict(X_test))
    
    return(MAE)



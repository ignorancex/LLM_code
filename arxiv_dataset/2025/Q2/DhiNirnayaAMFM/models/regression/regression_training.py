# -*- coding: utf-8 -*-
"""
Regression Model Training Script

@author: seema

Description:
This script performs regression using two machine learning models: 
Decision Tree and Support Vector Machine (SVM). It includes hyperparameter 
tuning with GridSearchCV, evaluation on training and test datasets, 
and saves both model objects and evaluation metrics.

Inputs:
- Train CSV file (with features and 'mmse', 'dx' columns)
- Test CSV file (same structure as train)
- User-defined working directory for saving models and results

Outputs:
- Trained model files (.pkl)
- Evaluation metrics (CSV summary)

"""

import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import pearsonr

#%%
# Define regression models and hyperparameter grids for tuning
models = {
    "DecisionTree": (DecisionTreeRegressor(), {"max_depth": [3, 5, 7, 10, 15, 17, 20, 25, 50, None],
                                               "min_samples_split": [2, 5, 7, 10, 12, 15],
                                               "min_samples_leaf": [1, 2, 5, 7, 10, 13, 15, 17, 20, 23, 25, 30],
                                               'max_features': ['auto', 'sqrt', 'log2', None]}),
    "SVM": (SVR(), {"C": [0.1, 0.5, 1, 3, 5, 10, 13, 15], 
                    "gamma": [0.001, 0.01, 0.1, "scale"], 
                    "kernel": ["rbf", "linear", "poly"]})
}

#%%
# Define the directory to store trained models and summary results
working_dir = "<Path/to/working/directory>"

# Check if the directory exists, if not, create it
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# Switch to working directory
os.chdir(working_dir)

#%%
# Set the paths to training and testing CSV files
train_ip_file_path = "<Path/to/train/csv>"
test_ip_file_path = "<Path/to/test/csv>"           
        
# Read training data
train_df = pd.read_csv(train_ip_file_path, index_col=0)
# Shuffle the training data for better generalization
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Separate features and target variable
X_train = train_df.drop(columns=['mmse', 'dx'])
y_train = train_df['mmse']     

# REad testing data
test_df = pd.read_csv(test_ip_file_path, index_col=0)
# Separate features and target variable for test set
X_test = test_df.drop(columns=['mmse', 'dx'])
y_test = test_df['mmse']     

#%%
# Standardize features using StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

#%%
# Initialize the results CSV with header
summary_path = os.path.join(working_dir, "regression_summary.csv")
pd.DataFrame([["model_name", 
               "r-value train", "RMSE train", "r2 train", "MSE train", "MAE train",
               "r-value test", "RMSE test", "r2 test", "MSE test", "MAE test"]]).to_csv(summary_path, header=False, index=False)

#%% Iterate through each model for training, evaluation, and saving results
for model_name, (model, param_grid) in models.items():
    print(f"Training {model_name}")
    start_time = time.time()

    # Perform grid search if hyperparameters are defined
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Params: {grid_search.best_params_}")
    else:
        best_model = model.fit(X_train_scaled, y_train)

    elapsed_time = time.time() - start_time
    print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")

    # Save the trained model
    model_path = os.path.join(working_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Evaluate on training data
    y_train_pred = best_model.predict(X_train_scaled)            
    r_val_train, _ = pearsonr(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))            
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)

    # Evaluate on testing data 
    y_test_pred = best_model.predict(X_test_scaled)            
    r_val_test, _ = pearsonr(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))            
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"{model_name} - R-value: {r_val_test:.2f}, MSE: {rmse_test:.2f}")

    # Append evaluation metrics to summary CSV    
    pd.DataFrame([[model_name, 
                   r_val_train, rmse_train, r2_train, mse_train, mae_train,
                   r_val_test, rmse_test, r2_test, mse_test, mae_test]]).to_csv(summary_path, mode='a', header=False, index=False)
#%%
print("Training and evaluation completed.")

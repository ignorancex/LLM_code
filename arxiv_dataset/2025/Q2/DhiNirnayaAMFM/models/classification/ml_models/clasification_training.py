# -*- coding: utf-8 -*-
"""
Classification Model Training Script

@author: seema

Description:
This script performs classification using Decision Tree and Support Vector Machine (SVM) classifiers.
It supports hyperparameter tuning using GridSearchCV, label encoding, model saving, and evaluation.
Results are generated for different input datasets, feature types, and DCT settings.
"""


import os
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#%%
# Define classification models and hyperparameter search space
models = {
    "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 7, 10, 15, None], 
                                                "min_samples_split": [2, 5, 10], 
                                                "min_samples_leaf": [1, 5, 10], 
                                                "max_features": ['sqrt', 'log2', None]}),
    "SVM": (SVC(probability=True), {"C": [0.1, 0.5, 1, 3, 5, 10], 
                    "kernel": ["linear", "rbf", "poly", "sigmoid"], 
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1], 
                    "degree": [2, 3, 4]})
}

#%%
# Set the working directory where results will be saved
working_dir = "<Path/to/working/directory>" 

# Check if the directory exists, if not, create it
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# Switch to the working directory
os.chdir(working_dir)

#%% Define the paths to the training and testing datasets
train_ds_path = "<Path/to/train/csv>"
test_ds_path =  "<Path/to/test/csv>"

#%%
# Specify the file indices and feature types to iterate over
file_indices = [2, 3, 4]
feat_set = ["variance", "ddct", "full_features"]


#%% # Loop over different dataset configurations
for index in file_indices:
    for feat_type in feat_set:
                
        # Construct paths to train and test files
        train_ip_file_path = os.path.join(train_ds_path, f"training_dataset_{feat_type}_dct_num_{index}.csv") 
        print(f"training_dataset_{feat_type}_dct_num_{index}.csv")
        
        test_ip_file_path = os.path.join(test_ds_path, f"testing_dataset_{feat_type}_dct_num_{index}.csv") 
        print(f"testing_dataset_{feat_type}_dct_num_{index}.csv")                  
        
        #%% Load train data
        train_df = pd.read_csv(train_ip_file_path)
        # Shuffle dataset
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        X_train = train_df.drop(columns=['mmse', 'dx'])
        y_train = train_df['dx']             
                
        # Load test data
        test_df = pd.read_csv(test_ip_file_path)
        X_test = test_df.drop(columns=['mmse', 'dx'])
        y_test = test_df['dx']  

        #%%
        # Scale features using StandardScaler
        scaler = StandardScaler()
        
        # Fit on training data and transform
        X_train_scaled = scaler.fit_transform(X_train) 
        X_test_scaled = scaler.transform(X_test) 

        # Encode categorical labels to numeric
        # Initialize the encoder
        label_encoder = LabelEncoder()
        # Fit and transform labels for training data
        y_train_encoded = label_encoder.fit_transform(y_train)        
        # Transform labels for test data (using same encoder)
        y_test_encoded = label_encoder.transform(y_test)
        
        #%% Create result directory for current dataset configuration
        results_dir = f"dct_num_{index}/feat_type_{feat_type}"
        os.makedirs(results_dir, exist_ok=True)
        print(f'Dct num {index}')
        print(f'Feat type {feat_type}')       
        
        # Create CSV summary file for current config
        summary_path = os.path.join(results_dir, "classification_summary.csv")
        
        pd.DataFrame([["index", "feat_type", "model_name", 
                       "acc_train", "prec_train", "rec_train", "f1_train",
                       "acc_test", "prec_test", "rec_test", "f1_test"]]).to_csv(summary_path, header=False, index=False)

        # Train and evaluate each model
        for model_name, (model, param_grid) in models.items():
            print(f"Training {model_name} for dct_num {index}, feat_type {feat_type}")
            start_time = time.time()

            # Perform Grid Search 
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X_train_scaled, y_train_encoded)
                best_model = grid_search.best_estimator_
                print(f"Best Params: {grid_search.best_params_}")
            else:
                best_model = model.fit(X_train_scaled, y_train_encoded)

            elapsed_time = time.time() - start_time
            print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")

            # Save the trained model
            model_path = os.path.join(results_dir, f"{model_name}_model_dct_{index}_feat_{feat_type}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

            # Evaluate on training set
            y_train_pred = best_model.predict(X_train_scaled) 
            acc_train = accuracy_score(y_train_encoded, y_train_pred)
            prec_train = precision_score(y_train_encoded, y_train_pred)
            rec_train = recall_score(y_train_encoded, y_train_pred)
            f1_train = f1_score(y_train_encoded, y_train_pred)
            
            # Evaluate on test set
            y_test_pred = best_model.predict(X_test_scaled) 
            acc_test = accuracy_score(y_test_encoded, y_test_pred)
            prec_test = precision_score(y_test_encoded, y_test_pred)
            rec_test = recall_score(y_test_encoded, y_test_pred)
            f1_test = f1_score(y_test_encoded, y_test_pred)
            
            print(f"{model_name} - Accuracy: {acc_test:.4f}")
            
            # Save metrics to summary file
            pd.DataFrame([[index, feat_type, model_name, 
                           acc_train, prec_train, rec_train, f1_train,
                           acc_test, prec_test, rec_test, f1_test]]).to_csv(summary_path, mode='a', header=False, index=False)

print("Training and evaluation completed.")

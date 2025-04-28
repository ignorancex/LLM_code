# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:52:28 2023

@author: David J. Kedziora
"""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
import time

dir_data = "../data/ml"

list_num_events = [None,
                   "1000",
                   "10000",
                   "100000",
                   "1000000",
                   None]

experimental_contexts = [None,
                         "1p2uW_3000cps",
                         "2p5uW_4000cps",
                         "4uW_4100cps",
                         "8uW_5100cps",
                         "10uW_6000cps",
                         "10uW_12000cps",
                         "20uW_7000cps",
                         "30uW_7000cps",
                         None]

model_types = [None,
            #    "dummy",
            #    "linear_regression",
               "linear_svr",
               None]

keys_drop = ["events", "estimate", "best"]

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(X_train, y_train, model_type):
    if model_type == "dummy":
        model = DummyRegressor()
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "linear_svr":
        model = LinearSVR()
    else:
        raise NotImplementedError
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared = False)
    return rmse

for model_type in model_types:
    
    if not model_type is None:
        for key_target in ["estimate", "best"]:

            loss = dict()

            for num_events in list_num_events:
                if not num_events is None:
                    loss[num_events] = dict()
                    for experimental_context in experimental_contexts:
                        if not experimental_context is None:
                            loss[num_events][experimental_context] = dict()

                            print(f"Training for '{key_target}' on num_events '{num_events}' and everything but '{experimental_context}'.")

                            # Load training data from all other experimental contexts
                            start_time = time.time()
                            train_data_concat = pd.DataFrame()
                            for other_context in experimental_contexts:
                                if other_context != experimental_context and other_context is not None:
                                    train_data_file = "%s/train_sps_quality_%s_events_%s.csv" % (dir_data, num_events, other_context)
                                    train_data = load_data(train_data_file)
                                    train_data_concat = pd.concat([train_data_concat, train_data])
                            train_loading_time = time.time() - start_time

                            # Load testing data for current num_events
                            for num_events_test in list_num_events:
                                if num_events_test is not None:
                                    print(f"Testing for '{key_target}' on num_events '{num_events_test}' and '{experimental_context}'.")
                                    test_data_file = "%s/test_sps_quality_%s_events_%s.csv" % (dir_data, num_events_test, experimental_context)
                                    start_time = time.time()
                                    test_data = load_data(test_data_file)
                                    test_loading_time = time.time() - start_time

                                    # Train the model
                                    X_train = train_data_concat.drop(keys_drop, axis=1)  # Adjust 'target_column_name'
                                    y_train = train_data_concat[key_target]  # Adjust 'target_column_name'
                                    model, training_time = train_model(X_train, y_train, model_type)

                                    # Test the model
                                    X_test = test_data.drop(keys_drop, axis=1)  # Adjust 'target_column_name'
                                    y_test = test_data[key_target]  # Adjust 'target_column_name'
                                    rmse_ml = test_model(model, X_test, y_test)

                                    if key_target == "best":
                                        rmse_fitting = mean_squared_error(y_test, test_data["estimate"], squared = False)
                                    else:
                                        rmse_fitting = None

                                    rmse_tuple = (rmse_ml, rmse_fitting)

                                    loss[num_events][experimental_context][num_events_test] = rmse_tuple

                                    print(f"Num Events: {num_events}, Num Events Test: {num_events_test}, Experimental Context: {experimental_context}")
                                    print(f"Training/testing rows: {train_data_concat.shape[0]}/{test_data.shape[0]}")
                                    print(f"Training Time: {training_time} seconds")
                                    print(f"Training Data Loading Time: {train_loading_time} seconds")
                                    print(f"Testing Data Loading Time: {test_loading_time} seconds")
                                    if key_target == "best":
                                        print(f"Root Mean Squared Error (Fitting): {rmse_fitting}")
                                    print(f"Root Mean Squared Error (ML): {rmse_ml}\n")

            # Flatten the nested dictionary into a list of dictionaries
            flattened_data = []
            for num_events, contexts in loss.items():
                for context, test_data in contexts.items():
                    for num_events_test, rmse_tuple in test_data.items():
                        temp_dict = {
                            "num_events": num_events,
                            "experimental_context": context,
                            "num_events_test": num_events_test,
                            "rmse_ml": rmse_tuple[0]
                        }
                        if key_target == "best":
                            temp_dict["rmse_fitting"] = rmse_tuple[1]
                        flattened_data.append(temp_dict)

            # Create a DataFrame from the flattened data
            df = pd.DataFrame(flattened_data)

            # Save the DataFrame to a CSV file
            csv_filename = "loss_for_%s_%s.csv" % (key_target, model_type)  # Specify the desired CSV filename
            df.to_csv(csv_filename, index=False)

            print(f"CSV file '{csv_filename}' has been saved.")
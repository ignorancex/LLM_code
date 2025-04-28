# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:25:21 2024

@author: Pietro Colombo
"""


import pandas as pd
import os
import numpy as np
from IPython.display import display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Display the DataFrame

# Specify the directory containing the files
directory = 'C:/Users/2692812C/OneDrive - University of Glasgow/Desktop/Projects/Glasgow-Hurricanes-main/Glasgow-Hurricanes-main/4__forecast_andComparingModel'
D = pd.read_csv(os.path.join(directory, "D_python.csv"))

# Initialize variables to store results
f_lstm = []
f_lstm_NOPP = []
test_target = []


# Define the range of years for testing
test_years = range(2011, 2022)

# Loop for 100 iterations
for _ in range(100):
    f_lstm_iteration = []  # List to store predictions for this iteration

    # Iterate over test years
    for test_year in test_years:
        # Subset the data for training and testing
        train_data = D[D['YYYY'] < test_year]
        test_data = D[D['YYYY'] == test_year+1]

        ##################
        # Data preparation
        ##################

        # Define input features and target variable for LSTM
        X_train = train_data[['NINA1_std_t_1', 'z500_std_t_1']].values
        y_train = train_data['target'].values

        X_test_df = test_data[['NINA1_std_t_1', 'z500_std_t_1']]  # Selecting columns from DataFrame
        X_test = X_test_df.values

        # Reshape input data for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Correct reshaping

        ##################
        # LSTM model
        ##################
        np.random.seed(42)
        # Define the model architecture
        model = Sequential([
            LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)[0][0]
        f_lstm_iteration.append(y_pred)

        # Store the true target value
        test_target.append(test_data['target'].values[-1])

    # Append predictions of this iteration to the matrix
    f_lstm.append(f_lstm_iteration)

# Convert f_lstm to numpy array
f_lstm = np.array(f_lstm)

# Print the shape of the prediction matrix
print("Shape of prediction matrix:", f_lstm.shape)

column_means = np.mean(f_lstm, axis=0)
column_sd = np.var(f_lstm, axis=0)

# Print column-wise means
print("Column-wise means:", column_means)



file_path = 'output_LSTM_NOPP.csv'
# Write the NumPy array to a CSV file
np.savetxt(file_path, f_lstm, delimiter=',')
print("CSV file saved successfully.")

test_target[0:10]


from sklearn.metrics import mean_absolute_error

def directional_accuracy(forecast, actual):

    """
    Compute directional accuracy between forecast and actual values.
    
    Parameters:
    - forecast (list): List of forecasted values.
    - actual (list): List of actual values.
    
    Returns:
    - da (float): Directional accuracy.
    """
    forecast_diff = np.diff(forecast)
    actual_diff = np.diff(actual)
    correct_directions = sum(np.sign(forecast_diff) == np.sign(actual_diff))
    da = correct_directions / (len(actual) - 1)
    return da

mae = mean_absolute_error(test_target[0:11], column_means[0:11])
print("Mean Absolute Error (MAE):", mae)

# Compute directional accuracy
da = directional_accuracy(column_means[0:11], test_target[0:11])
print("Directional Accuracy:", da)
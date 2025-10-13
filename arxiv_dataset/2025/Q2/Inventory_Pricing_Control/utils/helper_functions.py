# utils/helper_functions.py

import pandas as pd
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics.pairwise import pairwise_kernels
import itertools


def prepare_data(df):
    df = df.copy()
    df['Censor_1'] = df['Censor_1'].apply(lambda x: True if x == 1 else False)
    return df


def calculate_survival(df, x):
    event_occurred = df["Censor_1"]
    durations = df["Sales_1"]
    level, survival_prob = kaplan_meier_estimator(event_occurred, durations)
    survival_df = pd.DataFrame({'level': level, 'survival_prob': survival_prob})
    con_survival = survival_df.loc[survival_df['level'] == x, 'survival_prob'].values[0]
    integ_survival = survival_df.loc[survival_df['level'] >= x, 'survival_prob'].sum()
    return con_survival, integ_survival


def estimate_demand(x, df):
    prepared_df = prepare_data(df[['Censor_1', 'Sales_1']])
    con_survival, integ_survival = calculate_survival(prepared_df, int(x))
    estimated_demand = int(x) + (1 / con_survival) * integ_survival
    return estimated_demand


def impute_reward(df):
    df_1 = df.copy()
    df_1["Estimated Demand"] = df_1["Demand"]
    df_1["Imputed Profit"] = df_1["Observed Profit"]
    indexes = df_1.loc[df_1["Censor_1"] == 0].index.to_list()
    df_1.loc[indexes, "Estimated Demand"] = df_1.loc[indexes].apply(
        lambda row: estimate_demand(row["Sales_1"], df), axis=1)
    df_1.loc[indexes, "Imputed Profit"] = df_1.loc[indexes].apply(
        lambda row: row["Observed Profit"] - 2 * (row["Estimated Demand"] - row["Sales_1"]), axis=1)
    return df_1


def expand_and_predict_max(df, ordering, pricing, model):
    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
    df = df.reset_index()

    df['key'] = 1
    grid['key'] = 1

    expanded_df = pd.merge(df, grid, on='key').drop('key', axis=1)

    expanded_df.columns = ["index"] + list(model.feature_names_in_)
    predictions = model.predict(expanded_df.iloc[:, 1:])
    expanded_df['predictions'] = predictions

    max_predictions = expanded_df.groupby(["index"])['predictions'].max()

    return max_predictions


def expand_and_predict_max_pess(df, ordering, pricing, model, index_dataset):
    grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
    df = df.reset_index()

    df['key'] = 1
    grid['key'] = 1

    expanded_df = pd.merge(df, grid, on='key').drop('key', axis=1)

    expanded_df.columns = ["index"] + list(model.feature_names_in_)
    predictions = model.predict(expanded_df.iloc[:, 1:])

    # Compute uncertainty quantification
    UQ = compute_sigma_for_dataframe(
        expanded_df.iloc[:, 1:],
        df.iloc[:, :len(model.feature_names_in_)],
        kernel=model.get_params()['kernel_ridge__kernel'],
        gamma=model.get_params()['kernel_ridge__gamma'],
        zeta=model.get_params()['kernel_ridge__alpha'],
        beta_no=1
    )

    expanded_df['predictions'] = predictions - UQ

    max_predictions = expanded_df.groupby(["index"])['predictions'].max()

    return max_predictions


def compute_sigma_for_dataframe(X, x_no, kernel='rbf', gamma=1.0, zeta=2.0, beta_no=1.0):
    K_no = pairwise_kernels(x_no, x_no, metric=kernel, gamma=gamma)
    inverse_term = np.linalg.inv(K_no + zeta * np.eye(x_no.shape[0]))

    def compute_single_sigma(x):
        k_no_x = pairwise_kernels(x_no, x.reshape(1, -1), metric=kernel, gamma=gamma)
        k_no_x_prime_x = pairwise_kernels(x.reshape(1, -1), x.reshape(1, -1), metric=kernel, gamma=gamma) - \
                         k_no_x.T @ inverse_term @ k_no_x
        sigma_x = beta_no * (k_no_x_prime_x / zeta)
        return sigma_x[0, 0]

    sigma_values = X.apply(lambda row: compute_single_sigma(row.values), axis=1)
    return sigma_values


def generate_partition(data, i):
    if i < 1:
        raise ValueError("i must be at least 1")

    for j in range(-i, 2):
        data[f'Inv_{j}'] = data['Inv_0'].shift(-j)
        data[f'Censor_{j}'] = data['Censor_0'].shift(-j)
        data[f'Sales_{j}'] = data['Sales_0'].shift(-j)
        data[f'State_{j}'] = data['State_0'].shift(-j)
        data[f'IRate_{j}'] = data['IRate_0'].shift(-j)
        if j != 1:
            data[f'Ordering_{j}'] = data['Ordering_0'].shift(-j)
            data[f'Pricing_{j}'] = data['Pricing_0'].shift(-j)
    conditions = (data[f'Censor_0'] == 0)
    for j in range(1, i):
        conditions &= (data[f'Censor_{-j}'] == 0)
    conditions &= (data[f'Censor_{-i}'] == 1)
    filtered_data = data[conditions]
    columns = []
    for j in range(-i, 2):
        columns.extend([f'Inv_{j}', f'Censor_{j}', f'Sales_{j}', f'State_{j}', f'IRate_{j}'])
        if j != 1:
            columns.extend([f'Ordering_{j}', f'Pricing_{j}'])
    columns.append('Imputed Profit')
    return filtered_data[columns]



def generalized_stack(env, log_names, counter, n=1):
    reshaped_logs = []

    # Iterate over each time step from 'counter-n' to 'counter'
    for j in range(counter - n, counter + 1):
        for log_name in log_names:
            log = getattr(env, log_name)  # Access the log from the env object

            if j < counter:  # Process as usual until the last step
                if log_name == 'state_log':
                    # Since state_log returns a 2D array (5x1), flatten this to a single row
                    log_entry = log[j].reshape(1,-1)
                else:
                    # For scalar logs, wrap the scalar in an array and flatten
                    log_entry = np.array([log[j]]).reshape(1,-1)

                # Append the processed log entry
                reshaped_logs.append(log_entry)
            else:  # When j == counter, handle only state_log
                if log_name == 'state_log':
                    # Handle only state_log at the last index
                    log_entry = log[j].reshape(1,-1)  # Flatten to ensure it is a single row
                    reshaped_logs.append(log_entry)
                # If log_name is not 'state_log', skip appending for this iteration

    # Stack all the reshaped logs horizontally
    result = np.hstack(reshaped_logs)
    return pd.DataFrame(result).reset_index()

def action_map(x):
    if x.iloc[1] == 2:
        return np.array([x.iloc[0], 0])
    elif x.iloc[1] == 4:
        return np.array([x.iloc[0], 1])
    else:
        return np.array([x.iloc[0], 2])


def count_zeros_from_last(data):
    data_list = list(data)
    try:
        last_one_index = len(data_list) - 1 - data_list[::-1].index(1)
        return data_list[last_one_index + 1:].count(0)
    except ValueError:
        return len(data_list)

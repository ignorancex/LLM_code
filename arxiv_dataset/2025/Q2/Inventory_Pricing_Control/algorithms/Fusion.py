# algorithms/combined_algorithms.py

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from envs.inv_pricing_management_env import InvPricingManagementMasterEnv
from utils.helper_functions import (
    expand_and_predict_max,
    expand_and_predict_max_pess,
    compute_sigma_for_dataframe,
    generate_partition,
    generalized_stack,
    action_map,
    count_zeros_from_last
)
import itertools


def run_fusion_algorithm(trained_on_optimal):
    """
    Runs the combined FQI and Pessimistic FQI algorithms using pre-existing datasets.
    """
    allowed_c = 3
    pricing = [2, 4, 6]
    log_names = ['state_log', 'ordering_log', 'price_log']

    rewards_dic = {}
    rewards_dic_pess = {}
    pricing = [2, 4, 6]

    for n_episode in range(10, 60, 10):
        for ind_run in range(50):
            print(f"Running episodes: {n_episode}, Run index: {ind_run}")
            env = InvPricingManagementMasterEnv(allowed_censoring=allowed_c)
            ordering = list(range(env.supply_capacity + 1))

            # Load dataset
            if trained_on_optimal==True:
                df1 = pd.read_excel(f'data/data_ind_{ind_run}_episode_{n_episode}.xlsx')
                df1.drop(df1.columns[0], axis=1, inplace=True)
                df1 = env.impute_reward(df1)
            else:
                df1 = env.data_generator(None, n_episode)
                df1 = env.impute_reward(df1)

            n = allowed_c + 1
            models = [np.nan for i in range(n)]
            gamma = 0.9
            O0_N = df1[df1["Censor_0"] == 1].drop(["Observed Profit", "Real Profit", "Demand", "Estimated Demand"],
                                                  axis=1)
            datasets = [O0_N if i == 0 else generate_partition(df1, i).dropna() for i in range(n)]

            pipelines = [Pipeline([
                ('scaler', StandardScaler()),
                ('kernel_ridge', KernelRidge())
            ]) for _ in range(n)]

            param_grid = {
                'kernel_ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel_ridge__kernel': ['rbf', 'laplacian'],
                'kernel_ridge__gamma': [0.01, 0.1, 1],  # Only relevant for rbf and poly

            }
            models_pess = [np.nan for i in range(n)]
            pipelines_pess = [Pipeline([
                ('scaler', StandardScaler()),
                ('kernel_ridge', KernelRidge())
            ]) for _ in range(n)]
            cv = KFold(n_splits=5)
            for i in range(10):  # Number of iterations
                for j in range(n):  # Iterate over each model
                    # print(i,j)
                    data = datasets[j]  # Get the dataset for the j-th model
                    if i == 0:
                        target = data.loc[:, "Imputed Profit"]
                    else:
                        rewards = data.loc[:, "Imputed Profit"]
                        index_model_1 = data["Censor_1"] == 0
                        index_model_0 = data["Censor_1"] == 1
                        if j < n - 1:
                            if i < 4:
                                combined_predictions = pd.Series(0, index=rewards.index)

                                if len(data.iloc[np.where(index_model_1)[0], :-1]) == 0:
                                    # Do nothing for model 1
                                    pass
                                else:
                                    # Perform the prediction for model 1
                                    predictions_model_1 = expand_and_predict_max_pess(
                                        data.iloc[np.where(index_model_1)[0], :-1],
                                        ordering,
                                        pricing,
                                        models[j + 1],
                                        j + 1
                                    )
                                    combined_predictions.loc[index_model_1] = predictions_model_1
                                if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                                    # Do nothing
                                    pass
                                else:
                                    # Perform the prediction
                                    predictions_model_0 = expand_and_predict_max_pess(
                                        data.iloc[np.where(index_model_0)[0], -6:-1],
                                        ordering,
                                        pricing,
                                        models[0],
                                        0
                                    )
                                    combined_predictions.loc[index_model_0] = predictions_model_0

                                target = rewards + gamma * combined_predictions

                            else:
                                combined_predictions = pd.Series(0, index=rewards.index)
                                if len(data.iloc[np.where(index_model_1)[0], :-1]) == 0:
                                    # Do nothing for model 1
                                    pass
                                else:
                                    # Perform the prediction for model 1
                                    predictions_model_1 = expand_and_predict_max(
                                        data.iloc[np.where(index_model_1)[0], :-1],
                                        ordering,
                                        pricing,
                                        models[j + 1]
                                    )
                                    combined_predictions.loc[index_model_1] = predictions_model_1
                                if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                                    # Do nothing
                                    pass
                                else:
                                    # Perform the prediction
                                    predictions_model_0 = expand_and_predict_max(
                                        data.iloc[np.where(index_model_0)[0], -6:-1],
                                        ordering,
                                        pricing,
                                        models[0]
                                    )
                                    combined_predictions.loc[index_model_0] = predictions_model_0

                                target = rewards + gamma * combined_predictions
                        else:
                            if i < 4:
                                combined_predictions = pd.Series(0, index=rewards.index)

                                if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                                    # Do nothing
                                    pass
                                else:
                                    # Perform the prediction
                                    predictions_model_0 = expand_and_predict_max_pess(
                                        data.iloc[np.where(index_model_0)[0], -6:-1],
                                        ordering,
                                        pricing,
                                        models[0],
                                        0
                                    )
                                    combined_predictions.loc[index_model_0] = predictions_model_0

                                target = rewards + gamma * combined_predictions



                            else:
                                combined_predictions = pd.Series(0, index=rewards.index)

                                if len(data.iloc[np.where(index_model_0)[0], -6:-1]) == 0:
                                    # Do nothing
                                    pass
                                else:
                                    # Perform the prediction
                                    predictions_model_0 = expand_and_predict_max(
                                        data.iloc[np.where(index_model_0)[0], -6:-1],
                                        ordering,
                                        pricing,
                                        models[0]
                                    )
                                    combined_predictions.loc[index_model_0] = predictions_model_0
                                target = rewards + gamma * combined_predictions

                    # Fit the model
                    k = 7 * (j + 1)

                    models[j] = GridSearchCV(pipelines[j], param_grid=param_grid, cv=cv,
                                             scoring="neg_mean_squared_error").fit(data.iloc[:, :k], target)
                    dic_param_pess = {'kernel_ridge__alpha': models[j].best_estimator_.get_params(['kernel_ridge'])[
                        'kernel_ridge__alpha'],
                                      'kernel_ridge__degree': models[j].best_estimator_.get_params(['kernel_ridge'])[
                                          'kernel_ridge__degree'],
                                      'kernel_ridge__gamma': models[j].best_estimator_.get_params(['kernel_ridge'])[
                                          'kernel_ridge__gamma'],
                                      'kernel_ridge__kernel': models[j].best_estimator_.get_params(['kernel_ridge'])[
                                          'kernel_ridge__kernel']}

                    pipelines[j].set_params(**dic_param_pess)
                    models[j] = pipelines[j].fit(data.iloc[:, :k], target)

            # Evaluate the models
            evaluate_models(env, models, models_pess, ordering, pricing, datasets, n_episode, ind_run, rewards_dic, rewards_dic_pess)

    # Save the results
    df_fqi = pd.DataFrame(rewards_dic)
    if trained_on_optimal==True:
        df_fqi.to_excel('outputs/output_fusion_optimal.xlsx', index=False)
    else:
        df_fqi.to_excel('outputs/output_fusion_uniform.xlsx', index=False)





def evaluate_models(env, models, models_pess, ordering, pricing, datasets, n_episode, ind_run, rewards_dic, rewards_dic_pess):
    """
    Evaluates the trained models by running simulations and collecting rewards.
    """
    gamma = 0.9
    eps = 100
    rewards_real_pess = []
    rewards_observed_pess = []
    censoring_observed_pess = []

    rewards_real = []
    rewards_observed = []
    censoring_observed = []

    # Pessimism Test
    for _ in range(eps):
        env_pess = InvPricingManagementMasterEnv(allowed_censoring=env.allowed_censoring)
        initial_state_pess = env_pess.reset()
        counter = 0
        while True:
            action_pess = select_action_pess(env_pess, models_pess, ordering, pricing, datasets, counter,initial_state_pess)
            _, _, done, _ = env_pess.step(action_pess)
            counter += 1
            if done:
                rewards_real_pess.append(env_pess.P.mean())
                rewards_observed_pess.append(env_pess.P_censored.mean())
                censoring_observed_pess.append(env_pess.Indicator_censor.mean())
                break
    rewards_dic_pess[f"{n_episode},{ind_run}"] = [
        np.mean(rewards_real_pess),
        np.mean(rewards_observed_pess),
        np.mean(censoring_observed_pess)
    ]

    # FQI Test
    for _ in range(eps):
        inital_state=env.reset()
        counter = 0
        while True:
            action = select_action(env, models, ordering, pricing, counter,inital_state)
            _, _, done, _ = env.step(action)
            counter += 1
            if done:
                rewards_real.append(env.P.mean())
                rewards_observed.append(env.P_censored.mean())
                censoring_observed.append(env.Indicator_censor.mean())
                break
    rewards_dic[f"{n_episode},{ind_run}"] = [
        np.mean(rewards_real),
        np.mean(rewards_observed),
        np.mean(censoring_observed)
    ]


def select_action(env, models, ordering, pricing, counter,inital_state):
    """
    Selects an action based on the current state and the trained models.
    """
    if counter == 0:
        state = pd.DataFrame(inital_state.reshape(1, -1)).reset_index()
        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
        expanded_df = pd.merge(state.assign(key=1), grid.assign(key=1), on='key').drop('key', axis=1)
        expanded_df.columns = ["index"] + list(models[0].feature_names_in_)
        Q_predicted = models[0].predict(expanded_df.iloc[:, 1:])
        action_taken = np.argmax(Q_predicted)
        action = action_map(grid.iloc[action_taken, 0:2])
    else:
        censor = count_zeros_from_last(env.Indicator_censor[:counter])
        states_actions = generalized_stack(env, ['state_log', 'ordering_log', 'price_log'], counter, censor)
        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
        expanded_df = pd.merge(states_actions.assign(key=1), grid.assign(key=1), on='key').drop('key', axis=1)
        expanded_df.columns = ["index"] + list(models[censor].feature_names_in_)
        Q_predicted = models[censor].predict(expanded_df.iloc[:, 1:])
        action_taken = np.argmax(Q_predicted)
        action = action_map(grid.iloc[action_taken, 0:2])
    return action


def select_action_pess(env_pess, models_pess, ordering, pricing, datasets, counter,initial_state_pess):
    """
    Selects an action for the pessimistic model based on the current state and the trained models.
    """
    if counter == 0:
        state = pd.DataFrame(initial_state_pess.reshape(1, -1)).reset_index()
        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
        expanded_df = pd.merge(state.assign(key=1), grid.assign(key=1), on='key').drop('key', axis=1)
        expanded_df.columns = ["index"] + list(models_pess[0].feature_names_in_)
        Q_predicted_pess = models_pess[0].predict(expanded_df.iloc[:, 1:])
        UQ = compute_uncertainty(expanded_df.iloc[:, 1:], models_pess[0], datasets[0])
        action_taken_pess = np.argmax(Q_predicted_pess - UQ)
        action_pess = action_map(grid.iloc[action_taken_pess, 0:2])
    else:
        censor = count_zeros_from_last(env_pess.Indicator_censor[:counter])
        states_actions_pess = generalized_stack(env_pess, ['state_log', 'ordering_log', 'price_log'], counter, censor)
        grid = pd.DataFrame(list(itertools.product(ordering, pricing)), columns=['Ordering_0', 'Pricing_0'])
        expanded_df_pess = pd.merge(states_actions_pess.assign(key=1), grid.assign(key=1), on='key').drop('key', axis=1)
        expanded_df_pess.columns = ["index"] + list(models_pess[censor].feature_names_in_)
        Q_predicted_pess = models_pess[censor].predict(expanded_df_pess.iloc[:, 1:])
        UQ = compute_uncertainty(expanded_df_pess.iloc[:, 1:], models_pess[censor], datasets[censor])
        action_taken_pess = np.argmax(Q_predicted_pess - UQ)
        action_pess = action_map(grid.iloc[action_taken_pess, 0:2])
    return action_pess


def compute_uncertainty(expanded_df, model, dataset):
    """
    Computes the uncertainty quantification for the pessimistic model.
    """
    kernel = model.get_params()['kernel_ridge__kernel']
    alpha = model.get_params()['kernel_ridge__alpha']
    gamma = model.get_params()['kernel_ridge__gamma']
    x_no = dataset.iloc[:, :len(model.feature_names_in_)].values

    UQ = compute_sigma_for_dataframe(
        expanded_df,
        x_no,
        kernel=kernel,
        gamma=gamma,
        zeta=alpha,
        beta_no=1
    )
    return UQ

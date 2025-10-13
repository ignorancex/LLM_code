# -*- coding: utf-8 -*-
"""
Synthetic Experiment: Continuous Treatment and Functional Outcomes

This script implements a synthetic experiment to evaluate causal function estimation methods with functional outcomes under a continuous treatment setting.
It simulates data, evaluates kernel-based causal effect estimators, and visualizes the results.

Key Components:
1. Data Generation:
   - `generate_synthetic_curves` simulates:
     - Outcome curves (Y), continuous treatment variable (D), and covariates (X, V).
     - Ground truth causal effects (theta_1) with Gaussian-like peaks perturbed based on D.
   - Supports both continuous and discrete treatment types with adjustable noise.

2. Model Evaluation:
   - **Kernel Ridge Regression Causal Estimator**:
     - Estimates Average Treatment Effects (ATE) and Conditional Average Treatment Effects (CATE).
     - Supports SRVF registration for aligning functional data (Y or V).
     - Allows scalar and operator-valued kernel-based estimation for multivariate outcomes.
   - Evaluation Functions:
     - `evaluate_model`: Scalar kernel estimator for ATE and CATE.
     - `evaluate_functional_model`: Operator-valued kernel estimator with SRVF options.
   - Metrics:
     - Mean Squared Error (MSE) between predicted and true ATE values.

3. Hyperparameter Optimization:
   - Performs grid search to optimize kernel ridge regression parameters:
     - Regularization parameters (lambda_1, lambda_2, lambda_3).
     - Bandwidth for Gaussian kernel (kernel_sigma).

4. Visualization:
   - `plot_combined_results`:
     - Boxplots of ATE errors across sample sizes and methods.
     - Line plots of dynamic errors as a function of time.
   - Outcome curves grouped by treatment bins (low, medium, high) for visualization.

5. Simulation Workflow:
   - Simulates data for different sample sizes (e.g., 50, 250, 500).
   - Evaluates three causal estimation methods:
     1. Standard Kernel Causal Estimator (Singh et al., 2023).
     2. Operator-Valued Kernel Estimator.
     3. SRVF-Registered Kernel Estimator.
   - Aggregates and visualizes error metrics (ATE errors, CATE errors) over multiple simulations.

6. Utilities:
   - **Nadaraya-Watson Regression**: Kernel-based regression for propensity scores or other relationships.
   - **SRVF Transformation**: Aligns curves to account for temporal misalignments.

Execution Instructions:
- Install Required Libraries: `numpy`, `matplotlib`, `seaborn`, `pandas`, `sklearn`, `fdasrsf`, `scipy`.
- Configure Parameters:
  - Update sample sizes, timepoints, hyperparameter ranges, and kernel parameters as needed.
- Run Script:
  - Generates synthetic data, evaluates models, and visualizes results.
  - Outputs include ATE and CATE error plots across methods and sample sizes.

Purpose:
This script benchmarks causal estimation methods for continuous treatments and functional outcomes, integrating SRVF registration and operator-valued kernels
to address temporal misalignments and complex dependencies in outcomes. If you use this code or any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.

@author: Yordan P. Raykov, Hengrui Luo 
"""

import numpy as np
import pandas as pd
from itertools import product

from utils import plot_combined_continious_treatment, generate_synthetic_curves, reshape_and_split_data
from utils import compute_median_interpoint_distance
from utils import evaluate_kernel_causal_effects, evaluate_func_kernel_causal_effects

from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator


## Evaluate the algorithms on synthetic data with continious treatment and functional outcome

frac_train = 0.80

Kernel_ATE_Error_Nsim = []
Kernel_functional_ATE_Error_Nsim = []
Kernel_SRVF_heterogeneous_ATE_Error_Nsim = []

Kernel_ATE_dynamic = []
Kernel_functional_ATE_dynamic = []
Kernel_SRVF_ATE_dynamic = []

Kernel_ATE_STD_Nsim = []
Kernel_functional_ATE_STD_Nsim = []
Kernel_SRVF_heterogeneous_ATE_STD_Nsim = []

# Initialize an empty DataFrame for combined data
combined_df = pd.DataFrame()

# Initialize a dictionary for dynamic DataFrames
dynamic_dfs = {}

for n_samples in [50, 100, 250]:
    n_samples_train = int(frac_train * n_samples)
    n_samples_test = n_samples - n_samples_train
    n_timepoints = 48
    num_sim = 20

    # Error storage

    Kernel_ATE_Error = []
    Kernel_heterogeneous_ATE_Error = []
    Kernel_heterogeneous_ATE_Error_values = []
    Kernel_V_values = []
    Kernel_ATE_Error_ATE_STD = []

    Kernel_functional_ATE_Error = []
    Kernel_functional_heterogeneous_ATE_Error = []
    Kernel_functional_heterogeneous_ATE_Error_values = []
    Kernel_functional_V_values = []
    Kernel_functional_ATE_Error_ATE_STD = []

    Kernel_SRVF_heterogeneous_ATE_Error = []
    Kernel_SRVF_heterogeneous_ATE_Error_values = []
    Kernel_SRVF_ATE_Error = []
    Kernel_SRVF_ATE_Error_STD = []
    Kernel_SRVF_V_values = []

    flag_ridge_penalties = 1
    flag_functional_ridge_penalties = 1
    flag_srvf_ridge_penalties = 1

    for _ in range(num_sim):

        # Generate synthetic data of outcomes, treatment, covariates and ground truth causal effect curves
        Y, D, V, X, theta_1_true = generate_synthetic_curves(n_samples, n_timepoints, treatment='continuous')
        # Split the data
        Y_train, D_train, X_train, V_train, theta_1_true_train, Y_test, D_test, X_test, V_test, theta_true_test = reshape_and_split_data(
            Y, D, V, X, theta_1_true, frac_train, random_state=42)
        kernel_sigma = compute_median_interpoint_distance(X_train)

        # Compute ind_v_test
        ind_v_test = np.argsort(V_test, axis=0).flatten()

        ##### First method - (1) Kernel Causal Effect Estimator from (Singh, Xu and Gretton 2023)

        if flag_ridge_penalties == 1: # Find optimal hyperparameters at the first simulation

            # Estimate the optimal ridge regression penalties lambda1, lambda2 and lambda3

            # Step 1: Define the hyperparameter search space for lambd1, lambd2 and lambd3
            # Define the grid of hyperparameters
            kernel__lambd1 = np.logspace(-3, 3, 3)
            kernel__lambd2 = np.logspace(-3, 3, 3)
            kernel__lambd3 = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1, kernel__lambd2, kernel__lambd3))

            # Step 2: Perform grid search with cross-validation
            best_score = float('inf')
            best_params = None

            for kernel__lambd1, kernel__lambd2, kernel__lambd3 in hyperparameter_grid:
                #theta_1_true[ind_v_test,:]
                score = evaluate_kernel_causal_effects(X_train, V_train, D_train, Y_train, X_test, V_test, D_test, Y_test, kernel__lambd1, kernel__lambd2, kernel__lambd3, kernel_sigma) # Evaluates ability to predict true theta1 on training set
                print(f"Evaluating kernel_lambd1={kernel__lambd1}, kernel_lambd2={kernel__lambd2}, kernel_lambd3 = {kernel__lambd3} MSE={score}")
                if score < best_score:
                    best_score = score
                    best_params = (kernel__lambd1, kernel__lambd2, kernel__lambd3)
            print(f"Best hyperparameters: kernel__lambd1={best_params[0]}, kernel__lambd2={best_params[1]}, kernel__lambd3={best_params[2]},  with MSE={best_score}")

            # Step 3: Extract the best lambda value and re-use them in simulations
            best_lambd1 = best_params[0]
            best_lambd2 = best_params[1]
            best_lambd3 = best_params[2]
            flag_ridge_penalties = 2 # no need to re-estimate best parameter values for the same n_samples

        kernel_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=best_lambd1, lambd2=best_lambd2, lambd3=best_lambd3, kernel_sigma = kernel_sigma, use_operator_valued_kernel=False)
        theta_ate_vector_per_t, theta_cate_vector_per_t = kernel_causal_estimator.fit(X_train, V_train, D_train, Y_train)

        predictions_ate, predictions_cate = kernel_causal_estimator.predict(X_test, V_test, D_test)

        theta_v_per_t_kernel = np.mean(predictions_cate[ind_v_test, :, :], axis=0)

        # Instatiate the current simulation error characteristics in estimating causal functions
        varphi_kernel_cate_error = theta_v_per_t_kernel - theta_true_test
        kernel_cate_error = np.mean(abs(varphi_kernel_cate_error[:, :]), axis=0)
        kernel_ate_error = np.mean(abs(np.mean(theta_true_test[:, :], axis=0) - np.mean(predictions_ate[:, :], axis=0)), axis=0)
        kernel_cate_error_std = np.std(abs(np.mean(varphi_kernel_cate_error[:, :], axis=0)))

        # Aggregate results in list
        Kernel_heterogeneous_ATE_Error.append(abs(np.mean(theta_true_test[:, :], axis=0) - np.mean(predictions_ate[:, :], axis=0)))
        Kernel_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_cate_error[ind_v_test, :]))
        Kernel_V_values.append(V_test[ind_v_test, :])
        Kernel_ATE_Error.append(kernel_ate_error)
        Kernel_ATE_Error_ATE_STD.append(kernel_cate_error_std)

        ##### Second method - (2) Naive operator valued Kernel Causal Effect Estimators - Section 4.1
        if flag_functional_ridge_penalties == 1: # Find optimal hyperparameters at the first simulation

            # Estimate the optimal ridge regression penalties lambda1, lambda2 and lambda3

            # Step 1: Define the hyperparameter search space for lambd1, lambd2 and lambd3
            # Define the grid of hyperparameters
            kernel__lambd1 = np.logspace(-3, 3, 3)
            kernel__lambd2 = np.logspace(-3, 3, 3)
            kernel__lambd3 = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1, kernel__lambd2, kernel__lambd3))

            # Step 2: Perform grid search with cross-validation
            best_score = float('inf')
            best_params = None

            for kernel__lambd1, kernel__lambd2, kernel__lambd3 in hyperparameter_grid:
                #theta_1_true[ind_v_test,:]
                score = evaluate_func_kernel_causal_effects(X_train, V_train, D_train, Y_train, X_test, V_test, D_test, Y_test, kernel__lambd1, kernel__lambd2, kernel__lambd3, kernel_sigma, apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False) # Evaluates ability to predict true theta1 on training set
                print(f"Evaluating kernel_lambd1={kernel__lambd1}, kernel_lambd2={kernel__lambd2}, kernel_lambd3 = {kernel__lambd3} MSE={score}")
                if score < best_score:
                    best_score = score
                    best_params = (kernel__lambd1, kernel__lambd2, kernel__lambd3)
            print(f"Best hyperparameters: kernel__lambd1={best_params[0]}, kernel__lambd2={best_params[1]}, kernel__lambd3={best_params[2]},  with MSE={best_score}")

            # Step 3: Extract the best lambda value and re-use them in simulations
            best_lambd1 = best_params[0]
            best_lambd2 = best_params[1]
            best_lambd3 = best_params[2]
            flag_functional_ridge_penalties = 2 # no need to re-estimate best parameter values for the same n_samples

        operator_kernel_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=best_lambd1, lambd2=best_lambd2, lambd3=best_lambd3, kernel_sigma = kernel_sigma, treatment='continuous', use_operator_valued_kernel=True, apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False)#, treatment_type='continious')
        theta_ate_vector_per_t, theta_cate_vector_per_t = operator_kernel_causal_estimator.fit(X_train, V_train, D_train, Y_train)
        predictions_operator_ate, predictions_operator_cate = operator_kernel_causal_estimator.predict(X_test, V_test, D_test)

        # Recover the indices for the treatment categories
        theta_v_per_t_kernel = np.mean(predictions_operator_cate[ind_v_test, :, :], axis=0)

        # Instatiate the current simulation error characteristics in estimating causal functions
        varphi_kernel_operator_cate_error = theta_v_per_t_kernel - theta_true_test
        kernel_cate_error = np.mean(abs(varphi_kernel_operator_cate_error[:, :]), axis=0)
        kernel_ate_error = np.mean(abs(np.mean(theta_true_test[:, :], axis=0) - np.mean(predictions_operator_ate[:, :], axis=0)), axis=0)
        kernel_operator_cate_error_std = np.std(abs(np.mean(varphi_kernel_operator_cate_error[:, :], axis=0)))

        # Aggregate results in list
        Kernel_functional_heterogeneous_ATE_Error.append(abs(np.mean(theta_true_test[:, :], axis=0) - np.mean(predictions_operator_ate[:, :], axis=0)))
        Kernel_functional_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_operator_cate_error[ind_v_test, :]))
        Kernel_functional_V_values.append(V_test[ind_v_test, :])
        Kernel_functional_ATE_Error.append(kernel_ate_error)
        Kernel_functional_ATE_Error_ATE_STD.append(kernel_operator_cate_error_std)

        ###### Third method - (3) Kernel Causal Effect Estimator with phase-shifts ######

        if flag_srvf_ridge_penalties == 1: # Find optimal hyperparameters at the first simulation

            # Estimate the optimal ridge regression penalties lambda1, lambda2 and lambda3

            # Step 1: Define the hyperparameter search space for lambd1, lambd2 and lambd3
            # Define the grid of hyperparameters
            kernel__lambd1 = np.logspace(-3, 3, 3)
            kernel__lambd2 = np.logspace(-3, 3, 3)
            kernel__lambd3 = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1, kernel__lambd2, kernel__lambd3))

            # Step 2: Perform grid search with cross-validation
            best_score = float('inf')
            best_params = None

            for kernel__lambd1, kernel__lambd2, kernel__lambd3 in hyperparameter_grid:
                score = evaluate_func_kernel_causal_effects(X_train, V_train, D_train, Y_train, X_test, V_test, D_test, Y_test, kernel__lambd1, kernel__lambd2, kernel__lambd3, kernel_sigma, apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False, srfv_X_groups=[1])
                print(f"Evaluating kernel_lambd1={kernel__lambd1}, kernel_lambd2={kernel__lambd2}, kernel_lambd3 = {kernel__lambd3} MSE={score}")
                if score < best_score:
                    best_score = score
                    best_params = (kernel__lambd1, kernel__lambd2, kernel__lambd3)
            print(f"Best hyperparameters: kernel__lambd1={best_params[0]}, kernel__lambd2={best_params[1]}, kernel__lambd3={best_params[2]},  with MSE={best_score}")

            # Step 3: Extract the best lambda value and re-use them in simulations
            best_srvf_lambd1 = best_params[0]
            best_srvf_lambd2 = best_params[1]
            best_srvf_lambd3 = best_params[2]
            flag_srvf_ridge_penalties = 2 # no need to re-estimate best parameter values for the same n_samples

        operator_kernel_SRVF_causal_estimator = KernelRidgeRegressionCausalEstimator(lambd1=best_srvf_lambd1, lambd2=best_srvf_lambd2, lambd3=best_srvf_lambd3, kernel_sigma = kernel_sigma, treatment='continuous', use_operator_valued_kernel=True, apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=True, srfv_X_groups=[1])#,  treatment_type='continious')  # Apply SRVF registration to the treated group)
        theta_srvf_ate_vector_per_t, theta_srvf_cate_vector_per_t = operator_kernel_SRVF_causal_estimator.fit(X_train, V_train, D_train, Y_train)
        predictions_ate_srvf, predictions_cate_srvf = operator_kernel_SRVF_causal_estimator.predict(X_test, V_test, D_test) # per covariate value

        # Recover the indices for the treatment categories
        theta_v_per_t_kernel_srvf = np.mean(predictions_cate_srvf[ind_v_test, :, :], axis=0)

        varphi_kernel_SRVF_cate_error = theta_v_per_t_kernel_srvf - theta_true_test # vector difference per covariate value
        kernel_SRVF_cate_error = np.mean(abs(varphi_kernel_SRVF_cate_error[:,:]), axis=0) # Average error across each value of the covariates (conditional ATE)
        kernel_SRVF_ate_error = np.mean(abs(np.mean(theta_true_test[:,:], axis=0) - np.mean(predictions_ate_srvf[ind_v_test, :], axis=0)), axis=0) # Average treatment effect (ATE)
        kernel_SRVF_cate_error_std = np.std(abs(np.mean(varphi_kernel_SRVF_cate_error[:,:], axis=0)))  

        Kernel_SRVF_heterogeneous_ATE_Error.append(abs(np.mean(theta_true_test[:, :], axis=0) - np.mean(predictions_ate_srvf[:, :], axis=0)))
        Kernel_SRVF_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_SRVF_cate_error[ind_v_test, :]))
        Kernel_SRVF_ATE_Error.append(kernel_SRVF_ate_error)
        Kernel_SRVF_ATE_Error_STD.append(kernel_SRVF_cate_error_std)

    Kernel_ATE_Error_Nsim.append(Kernel_ATE_Error)
    Kernel_functional_ATE_Error_Nsim.append(Kernel_functional_ATE_Error)
    Kernel_SRVF_heterogeneous_ATE_Error_Nsim.append(Kernel_SRVF_ATE_Error)

    Kernel_ATE_dynamic.append(np.mean(Kernel_heterogeneous_ATE_Error, axis=0))
    Kernel_functional_ATE_dynamic.append(np.mean(Kernel_functional_heterogeneous_ATE_Error, axis=0))
    Kernel_SRVF_ATE_dynamic.append(np.mean(Kernel_SRVF_heterogeneous_ATE_Error, axis=0))

    Kernel_ATE_STD_Nsim.append(np.mean(Kernel_ATE_Error_ATE_STD))
    Kernel_functional_ATE_STD_Nsim.append(np.mean(Kernel_functional_ATE_Error_ATE_STD))
    Kernel_SRVF_heterogeneous_ATE_STD_Nsim.append(np.mean(Kernel_SRVF_ATE_Error_STD))


    # Dataframes for plotting
    df = pd.DataFrame({
        'Kernel_ATE_Error': Kernel_ATE_Error,
        'Kernel_functional_ATE_Error': Kernel_functional_ATE_Error,
        'Kernel_SRVF_ATE_Error': Kernel_SRVF_ATE_Error
    })
    df['Sample_Size'] = n_samples
    combined_df = pd.concat([combined_df, df], ignore_index=True)


    df_dynamic = pd.DataFrame({
        'Kernel_ATE_dynamic': np.mean(Kernel_heterogeneous_ATE_Error, axis=0),
        'Kernel_functional_ATE_dynamic': np.mean(Kernel_functional_heterogeneous_ATE_Error, axis=0),
        'Kernel_SRVF_ATE_dynamic': np.mean(Kernel_SRVF_heterogeneous_ATE_Error, axis=0)
    })

    dynamic_dfs[n_samples] = df_dynamic

# Melt the combined DataFrame for seaborn's boxplot
melted_df = combined_df.melt(id_vars='Sample_Size', var_name='Method', value_name='Error')

# Plot combined results
plot_combined_continious_treatment(melted_df, dynamic_dfs)



# Define a dictionary mapping the old column names to new column names
rename_dict = {
    'Kernel_ATE_dynamic': 'Kernel DS',
    'Kernel_functional_ATE_dynamic': 'Operator Kernel DS',
    'Kernel_SRVF_ATE_dynamic': 'SRVF Operator Kernel DS'
}

rename_dict_error = {
    'Kernel_ATE_Error': 'Kernel DS',
    'Kernel_functional_ATE_Error': 'Operator Kernel DS',
    'Kernel_SRVF_ATE_Error': 'Iterative SRVF Kernel DS'
}
# Melt the combined DataFrame

# Rename the 'Method' column values
melted_df['Method'] = melted_df['Method'].replace(rename_dict_error)
# Create a new dictionary with renamed dataframes
dynamic_dfs_renamed = {}
for sample_size, df in dynamic_dfs.items():
    # Rename the columns using the rename_dict
    renamed_df = df.rename(columns=rename_dict)
    # Store the renamed dataframe in the new dictionary
    dynamic_dfs_renamed[sample_size] = renamed_df
    
plot_combined_continious_treatment(melted_df, dynamic_dfs_renamed)
# -*- coding: utf-8 -*-
"""
Synthetic Experiment: Functional Outcome Estimation with Causal Methods

This script evaluates various causal effect estimation methods in a functional outcomes context using synthetic data.
It incorporates classical and novel approaches, including SRVF (Square Root Velocity Function) registration for functional data alignment.

Key Features:
1. **Synthetic Data Generation**:
   - `generate_synthetic_curves`: Simulates functional outcomes (Y), covariates (X, V), treatment indicator (D), and true causal effects (theta_1_true).
   - Incorporates Gaussian-like peaks and perturbations in functional outcomes.

2. **Causal Effect Estimation Methods**:
   - Inverse Probability Weighting (IPW) with local smoothing.
   - Doubly Robust Estimation using kernel ridge regression for both propensity and outcome models.
   - Kernel-based Causal Effect Estimators:
     - Scalar Kernel Ridge Regression.
     - Operator-Valued Kernel for multivariate outcomes.
     - SRVF-based Kernel Estimation (applies functional data alignment).
   - Novel SRVF-registered IPW and Doubly Robust estimators.

3. **Evaluation Workflow**:
   - Split data into training and test sets using `reshape_and_split_data`.
   - Perform grid search for hyperparameter optimization:
     - Regularization parameters (\(\lambda_1, \lambda_2, \lambda_3\)).
     - Bandwidth (\(\sigma\)) for Gaussian kernel.
   - Evaluate methods on:
     - Average Treatment Effects (ATE).
     - Conditional Average Treatment Effects (CATE).
     - Heterogeneous ATE metrics.
   - Metrics include Mean Squared Error (MSE) and dynamic error profiles.

4. **SRVF Registration**:
   - Align functional data (outcomes Y and covariates V) using SRVF transformation via `fdasrsf`.
   - Evaluate causal methods both with and without SRVF preprocessing.

5. **Cross-Validation**:
   - `cross_validate_bandwidth`: Finds optimal kernel bandwidth for Nadaraya-Watson regression.
   - Grid search across penalty and kernel parameters for ridge regression.

6. **Visualization**:
   - `plot_combined_results`:
     - Boxplots for ATE errors across methods and sample sizes.
     - Line plots for dynamic errors over time.
   - Plots functional outcomes (Y) grouped by treatment.

7. **Implementation Details**:
   - Handles binary and continuous treatments.
   - Functional data is smoothed for SRVF registration using sliding windows.
   - Supports operator-valued kernels for multivariate outcomes.

8. **Results Aggregation**:
   - Aggregates errors and dynamic profiles for different methods across simulations.
   - Stores results in `combined_df` and `dynamic_dfs` for comparative analysis.

Execution Instructions:
- Install required libraries: `numpy`, `matplotlib`, `seaborn`, `pandas`, `sklearn`, `fdasrsf`, `scipy`.
- Adjust sample sizes, time points, and hyperparameter ranges as needed.
- Run script to simulate data, evaluate methods, and visualize performance.
- Outputs include visualizations of ATE errors, CATE errors, and functional outcome dynamics.

Purpose:
The script benchmarks classical and advanced causal estimation methods, including functional registration techniques, 
for functional outcomes under continuous treatment settings. It enables a deeper understanding of causal effects in scenarios 
where temporal misalignments and functional dependencies are critical. If you use this code or 
any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.

@author: Yordan P. Raykov, Hengrui Luo 
"""

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from itertools import product

from utils import plot_combined_binary_treatment, generate_synthetic_curves, reshape_and_split_data
from utils import nadaraya_watson, compute_median_interpoint_distance, cross_validate_bandwidth
from utils import evaluate_kernel_causal_effects, evaluate_func_kernel_causal_effects 

from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator


# Main experiment
frac_train = 0.80

IPW_ATE_Error_Nsim = []
Doubly_Robust_ATE_Error_Nsim = []
Kernel_ATE_Error_Nsim = []
Kernel_functional_ATE_Error_Nsim = []
Kernel_SRVF_heterogeneous_ATE_Error_Nsim = []

IPW_ATE_STD_Nsim = []
Doubly_Robust_ATE_STD_Nsim = []
Kernel_ATE_STD_Nsim = []
Kernel_functional_ATE_STD_Nsim = []
Kernel_SRVF_heterogeneous_ATE_STD_Nsim = []

IPW_ATE_dynamic = []
Doubly_Robust_ATE_dynamic = []
Kernel_ATE_dynamic = []
Kernel_functional_ATE_dynamic = []
Kernel_SRVF_ATE_dynamic = []

combined_df = pd.DataFrame()
dynamic_dfs = {}

for n_samples in [50, 100, 250]: 
    n_samples_train = int(frac_train * n_samples)
    n_samples_test = n_samples - n_samples_train
    n_timepoints = 48
    num_sim = 5

    IPW_heterogeneous_ATE_Error = []
    IPW_heterogeneous_ATE_Error_values = []
    IPW_V_values = []
    IPW_ATE_Error = []
    IPW_ATE_STD = []

    Doubly_Robust_ATE_Error = []
    Doubly_Robust_ATE_Error_values = []
    Doubly_Robust_ATE_STD = []
    Doubly_Robust_V_values = []
    Doubly_Robust_heterogeneous_ATE_Error = []

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
        # Generate data
        Y, D, V, X, theta_1_true = generate_synthetic_curves(n_samples, n_timepoints) # treatment_type='discrete', mode='fixed_x'
        # Split
        (Y_train, D_train, X_train, V_train, theta_1_true_train,
         Y_test, D_test, X_test, V_test, theta_1_true_test) = reshape_and_split_data(
            Y, D, V, X, theta_1_true, frac_train, random_state=42
        )
        kernel_sigma = compute_median_interpoint_distance(X_train)
        ind_v_test = np.argsort(V_test, axis=0).flatten()

        # --- (1) IPW ---
        varphi_ipw_ate = np.zeros((n_timepoints,))
        varphi_ipw_cate_error = np.zeros((n_samples_test, n_timepoints))

        # Get bandwidth via cross-validation
        bandwidths = np.linspace(0.01, 1, 5)
        optimal_bandwidth = cross_validate_bandwidth(X_train, D_train, bandwidths, k=5)

        # Propensity on test set
        propensity = nadaraya_watson(X_train, D_train, X_test, optimal_bandwidth)

        for t in range(n_timepoints):
            varphi_added = 0
            for i in range(n_samples_test):
                propensity_prob = propensity[i]
                # Observed "ipw" estimate for each test sample
                varphi = (
                    (D_test[i] * nadaraya_watson(X_train, Y_train[:, t], X_test[i:i+1, :], optimal_bandwidth) / propensity_prob)
                    - ((1-D_test[i]) * nadaraya_watson(X_train, Y_train[:, t], X_test[i:i+1, :], optimal_bandwidth) / (1 - propensity_prob))
                )
                # Store absolute error sample-by-sample
                varphi_ipw_cate_error[i, t] = abs(varphi - theta_1_true_test[i, t])
                varphi_added += varphi

            varphi_ipw_ate[t] = varphi_added / n_samples_test

        ipw_heterogeneous_ate_error = np.mean(varphi_ipw_cate_error, axis=0)
        ipw_ate_error = np.mean(abs(np.mean(theta_1_true_test, axis=0) - varphi_ipw_ate), axis=0)
        ipw_ate_error_std = np.std(abs(np.mean(theta_1_true_test, axis=0) - varphi_ipw_ate), axis=0)

        IPW_heterogeneous_ATE_Error.append(ipw_heterogeneous_ate_error)
        IPW_heterogeneous_ATE_Error_values.append(varphi_ipw_cate_error[ind_v_test, :])
        IPW_V_values.append(V_test[ind_v_test, :])
        IPW_ATE_Error.append(ipw_ate_error)
        IPW_ATE_STD.append(ipw_ate_error_std)


       #  ###### Second method - (2) Doubly Robust Estimator with kernel ridge regression model for propensity and conditional expectations ######

        # Index the treated/non-treated part of the training population
        ind_0_train = (D_train == 0).ravel()
        ind_1_train = (D_train == 1).ravel()
        # Estimate propensity using Kernel Ridge Regression
        propensity_model = KernelRidge(kernel='rbf', alpha=1.0, gamma=optimal_bandwidth)
        propensity_model.fit(X_train, D_train)  # Fit on train data
        propensity = propensity_model.predict(X_test)  # Predict on test data

        # Train separate outcome models for treated and untreated groups
        krr_0 = KernelRidge(kernel='rbf', alpha=1.0, gamma=optimal_bandwidth)
        krr_0.fit(X_train[ind_0_train], Y_train[ind_0_train])  # Model for untreated

        krr_1 = KernelRidge(kernel='rbf', alpha=1.0, gamma=optimal_bandwidth)
        krr_1.fit(X_train[ind_1_train], Y_train[ind_1_train])  # Model for treated

        # Compute Doubly Robust Estimates
        varphi_doubly_robust_cate_error = np.zeros((n_samples_test, n_timepoints))
        varphi_doubly_robust_ate = np.zeros((n_timepoints,))

        for t in range(n_timepoints):
            varphi_added = 0
            for i in range(n_samples_test):
                propensity_prob = np.clip(propensity[i], 1e-5, 1 - 1e-5)  # Avoid division by zero

                # Predict counterfactual outcomes
                y_0_pred = krr_0.predict(X_test[i:i+1])[:, t]
                y_1_pred = krr_1.predict(X_test[i:i+1])[:, t]

                # Compute DR estimator per sample
                varphi = ((D_test[i] * (Y_test[i, t] - y_1_pred) / propensity_prob) +
                  y_1_pred - ((1 - D_test[i]) * (Y_test[i, t] - y_0_pred) / (1 - propensity_prob)) +
                  y_0_pred)

                varphi_added += varphi
                varphi_doubly_robust_cate_error[i, t] = abs(varphi.item() - theta_1_true_test[i, t])  # Per sample-level error

            varphi_doubly_robust_ate[t] = varphi_added.item() / n_samples_test  # Compute average effect estimator

        # Compute errors consistently with other methods
        doubly_robust_cate_error = abs(np.mean(theta_1_true_test, axis=0) - np.mean(varphi_doubly_robust_cate_error, axis=0))
        doubly_robust_ate_error = np.mean(abs(np.mean(theta_1_true_test[:, :], axis=0) - np.mean(varphi_doubly_robust_cate_error, axis=0)), axis=0) 
        doubly_robust_ate_error_std = np.std(abs(np.mean(theta_1_true_test[:, :], axis=0) - np.mean(varphi_doubly_robust_ate, axis=0)), axis=0)  

        # Store results
        Doubly_Robust_ATE_Error.append(doubly_robust_ate_error)
        Doubly_Robust_ATE_Error_values.append(abs(varphi_doubly_robust_cate_error[ind_v_test, :]))
        Doubly_Robust_ATE_STD.append(doubly_robust_ate_error_std)
        Doubly_Robust_V_values.append(V_test[ind_v_test, :])
        Doubly_Robust_heterogeneous_ATE_Error.append(doubly_robust_cate_error)

        # --- (3) Kernel Causal Estimator (Scalar) ---
        if flag_ridge_penalties == 1:
            kernel__lambd1_vals = np.logspace(-3, 3, 3)
            kernel__lambd2_vals = np.logspace(-3, 3, 3)
            kernel__lambd3_vals = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1_vals, kernel__lambd2_vals, kernel__lambd3_vals))

            best_score = float('inf')
            best_params = None

            for lam1, lam2, lam3 in hyperparameter_grid:
                score = evaluate_kernel_causal_effects(
                    X_train, V_train, D_train, Y_train,
                    X_test, V_test, D_test, Y_test,
                    lam1, lam2, lam3, kernel_sigma
                )
                if score < best_score:
                    best_score = score
                    best_params = (lam1, lam2, lam3)
            best_lambd1, best_lambd2, best_lambd3 = best_params
            flag_ridge_penalties = 2
            print(f"Best hyperparams (scalar kernel): lambda1={best_lambd1}, lambda2={best_lambd2}, lambda3={best_lambd3}, MSE={best_score}")

        kernel_causal_estimator = KernelRidgeRegressionCausalEstimator(
            lambd1=best_lambd1, lambd2=best_lambd2, lambd3=best_lambd3,
            kernel_sigma=kernel_sigma, treatment = 'discrete', use_operator_valued_kernel=False
        )
        theta_ate_vector_per_t, theta_cate_vector_per_t = kernel_causal_estimator.fit(X_train, V_train, D_train, Y_train)
        predictions_ate, predictions_cate = kernel_causal_estimator.predict(X_test, V_test, D_test)

        ind_0_test = (D_test == 0).ravel()
        ind_1_test = (D_test == 1).ravel()

        theta_0_v_per_t_kernel = np.mean(predictions_cate[ind_0_test, :, :], axis=0)
        theta_1_v_per_t_kernel = np.mean(predictions_cate[ind_1_test, :, :], axis=0)
        theta_0_v_per_t_kernel = theta_0_v_per_t_kernel[ind_v_test, :]
        theta_1_v_per_t_kernel = theta_1_v_per_t_kernel[ind_v_test, :]

        varphi_kernel_cate_error = np.abs((theta_1_v_per_t_kernel - theta_0_v_per_t_kernel) - theta_1_true_test)
        kernel_cate_error = np.mean(varphi_kernel_cate_error, axis=0)
        kernel_ate_error = np.mean(abs(np.mean(theta_1_true_test, axis=0) - np.mean(predictions_ate, axis=0)), axis=0)
        kernel_cate_error_std = np.std(abs(np.mean(varphi_kernel_cate_error, axis=0)))

        Kernel_heterogeneous_ATE_Error.append(abs(np.mean(theta_1_true_test, axis=0) - np.mean(predictions_ate, axis=0)))
        Kernel_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_cate_error[ind_v_test, :]))
        Kernel_V_values.append(V_test[ind_v_test, :])
        Kernel_ATE_Error.append(kernel_ate_error)
        Kernel_ATE_Error_ATE_STD.append(kernel_cate_error_std)

        # --- (4) Naive operator valued Kernel Causal Effect Estimators  ---
        if flag_functional_ridge_penalties == 1:
            kernel__lambd1_vals = np.logspace(-3, 3, 3)
            kernel__lambd2_vals = np.logspace(-3, 3, 3)
            kernel__lambd3_vals = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1_vals, kernel__lambd2_vals, kernel__lambd3_vals))

            best_score = float('inf')
            best_params = None

            for lam1, lam2, lam3 in hyperparameter_grid:
                score = evaluate_func_kernel_causal_effects(
                    X_train, V_train, D_train, Y_train,
                    X_test, V_test, D_test, Y_test,
                    lam1, lam2, lam3, kernel_sigma,
                    apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False
                )
                if score < best_score:
                    best_score = score
                    best_params = (lam1, lam2, lam3)
            best_lambd1_func, best_lambd2_func, best_lambd3_func = best_params
            flag_functional_ridge_penalties = 2
            print(f"Best hyperparams (operator kernel): lambda1={best_lambd1_func}, lambda2={best_lambd2_func}, lambda3={best_lambd3_func}, MSE={best_score}")

        operator_kernel_causal_estimator = KernelRidgeRegressionCausalEstimator(
            lambd1=best_lambd1_func, lambd2=best_lambd2_func, lambd3=best_lambd3_func,
            kernel_sigma=kernel_sigma, treatment = 'discrete', use_operator_valued_kernel=True,
            apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False
        )
        theta_ate_vector_per_t, theta_cate_vector_per_t = operator_kernel_causal_estimator.fit(X_train, V_train, D_train, Y_train)
        predictions_operator_ate, predictions_operator_cate = operator_kernel_causal_estimator.predict(X_test, V_test, D_test)

        ind_0_test = (D_test == 0).ravel()
        ind_1_test = (D_test == 1).ravel()

        theta_0_v_per_t_kernel = np.mean(predictions_operator_cate[ind_0_test, :, :], axis=0)
        theta_1_v_per_t_kernel = np.mean(predictions_operator_cate[ind_1_test, :, :], axis=0)
        theta_0_v_per_t_kernel = theta_0_v_per_t_kernel[ind_v_test, :]
        theta_1_v_per_t_kernel = theta_1_v_per_t_kernel[ind_v_test, :]

        varphi_kernel_operator_cate_error = np.abs((theta_1_v_per_t_kernel - theta_0_v_per_t_kernel) - theta_1_true_test)
        kernel_cate_error = np.mean(varphi_kernel_operator_cate_error, axis=0)
        kernel_ate_error = np.mean(abs(np.mean(theta_1_true, axis=0) - np.mean(predictions_operator_ate, axis=0)), axis=0)
        kernel_operator_cate_error_std = np.std(abs(np.mean(varphi_kernel_operator_cate_error, axis=0)))

        Kernel_functional_heterogeneous_ATE_Error.append(abs(np.mean(theta_1_true_test, axis=0) - np.mean(predictions_operator_ate, axis=0)))
        Kernel_functional_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_operator_cate_error[ind_v_test, :]))
        Kernel_functional_V_values.append(V_test[ind_v_test, :])
        Kernel_functional_ATE_Error.append(kernel_ate_error)
        Kernel_functional_ATE_Error_ATE_STD.append(kernel_operator_cate_error_std)

        # --- (5) Kernel Causal Effect Estimator with phase-shifts ---
        if flag_srvf_ridge_penalties == 1:
            kernel__lambd1_vals = np.logspace(-3, 3, 3)
            kernel__lambd2_vals = np.logspace(-3, 3, 3)
            kernel__lambd3_vals = np.logspace(-3, 3, 3)
            hyperparameter_grid = list(product(kernel__lambd1_vals, kernel__lambd2_vals, kernel__lambd3_vals))

            best_score = float('inf')
            best_params = None

            for lam1, lam2, lam3 in hyperparameter_grid:
                score = evaluate_func_kernel_causal_effects(
                    X_train, V_train, D_train, Y_train,
                    X_test, V_test, D_test, Y_test,
                    lam1, lam2, lam3, kernel_sigma,
                    apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False
                )
                if score < best_score:
                    best_score = score
                    best_params = (lam1, lam2, lam3)
            best_srvf_lambd1, best_srvf_lambd2, best_srvf_lambd3 = best_params
            flag_srvf_ridge_penalties = 2
            print(f"Best hyperparams (SRVF): lambda1={best_srvf_lambd1}, lambda2={best_srvf_lambd2}, lambda3={best_srvf_lambd3}, MSE={best_score}")

        operator_kernel_SRVF_causal_estimator = KernelRidgeRegressionCausalEstimator(
            lambd1=best_srvf_lambd1, lambd2=best_srvf_lambd2, lambd3=best_srvf_lambd3,
            kernel_sigma=kernel_sigma, treatment = 'discrete', use_operator_valued_kernel=True,
            apply_srfv_Y=True, srfv_Y_groups=[1], apply_srfv_X=False
        )
        theta_srvf_ate_vector_per_t, theta_srvf_cate_vector_per_t = operator_kernel_SRVF_causal_estimator.fit(
            X_train, V_train, D_train, Y_train
        )
        predictions_ate_srvf, predictions_cate_srvf = operator_kernel_SRVF_causal_estimator.predict(
            X_test, V_test, D_test
        )

        ind_0 = (D_test == 0)
        ind_1 = (D_test == 1)
        theta_0_v_per_t_kernel_srvf = np.mean(predictions_cate_srvf[ind_0[:,0],:,:], axis=0)
        theta_1_v_per_t_kernel_srvf = np.mean(predictions_cate_srvf[ind_1[:,0],:,:], axis=0)
        theta_0_v_per_t_kernel_srvf = theta_0_v_per_t_kernel_srvf[ind_v_test,:]
        theta_1_v_per_t_kernel_srvf = theta_1_v_per_t_kernel_srvf[ind_v_test,:]

        varphi_kernel_SRVF_cate_error = np.abs(theta_1_v_per_t_kernel_srvf - theta_1_true_test)
        kernel_SRVF_cate_error = np.mean(varphi_kernel_SRVF_cate_error, axis=0)
        kernel_SRVF_ate_error = np.mean(abs(np.mean(theta_1_true_test, axis=0) - np.mean(predictions_ate_srvf[ind_v_test, :], axis=0)), axis=0)
        kernel_SRVF_cate_error_std = np.std(abs(np.mean(varphi_kernel_SRVF_cate_error, axis=0)))

        Kernel_SRVF_heterogeneous_ATE_Error.append(
            abs(np.mean(theta_1_true_test, axis=0) - np.mean(predictions_ate_srvf, axis=0))
        )
        Kernel_SRVF_heterogeneous_ATE_Error_values.append(abs(varphi_kernel_SRVF_cate_error[ind_v_test, :]))
        Kernel_SRVF_ATE_Error.append(kernel_SRVF_ate_error)
        Kernel_SRVF_ATE_Error_STD.append(kernel_SRVF_cate_error_std)

    IPW_ATE_Error_Nsim.append(np.mean(IPW_ATE_Error))
    Doubly_Robust_ATE_Error_Nsim.append(np.mean(Doubly_Robust_ATE_Error))
    Kernel_ATE_Error_Nsim.append(np.mean(Kernel_ATE_Error))
    Kernel_functional_ATE_Error_Nsim.append(np.mean(Kernel_functional_ATE_Error))
    Kernel_SRVF_heterogeneous_ATE_Error_Nsim.append(np.mean(Kernel_SRVF_ATE_Error))

    IPW_ATE_STD_Nsim.append(np.mean(IPW_ATE_STD))
    Doubly_Robust_ATE_STD_Nsim.append(np.mean(Doubly_Robust_ATE_STD))
    Kernel_ATE_STD_Nsim.append(np.mean(Kernel_ATE_Error_ATE_STD))
    Kernel_functional_ATE_STD_Nsim.append(np.mean(Kernel_functional_ATE_Error_ATE_STD))
    Kernel_SRVF_heterogeneous_ATE_STD_Nsim.append(np.mean(Kernel_SRVF_ATE_Error_STD))

    IPW_ATE_dynamic.append(np.mean(IPW_heterogeneous_ATE_Error, axis=0))
    Doubly_Robust_ATE_dynamic.append(np.mean(Doubly_Robust_heterogeneous_ATE_Error, axis=0))
    Kernel_ATE_dynamic.append(np.mean(Kernel_heterogeneous_ATE_Error, axis=0))
    Kernel_functional_ATE_dynamic.append(np.mean(Kernel_functional_heterogeneous_ATE_Error, axis=0))
    Kernel_SRVF_ATE_dynamic.append(np.mean(Kernel_SRVF_heterogeneous_ATE_Error, axis=0))

    # Collect into dataframes
    df = pd.DataFrame({
        'IPW_ATE_Error': IPW_ATE_Error,
        'Doubly_Robust_ATE_Error': Doubly_Robust_ATE_Error,
        'Kernel_ATE_Error': Kernel_ATE_Error,
        'Kernel_functional_ATE_Error': Kernel_functional_ATE_Error,
        'Kernel_SRVF_ATE_Error': Kernel_SRVF_ATE_Error
    })
    df['Sample_Size'] = n_samples
    combined_df = pd.concat([combined_df, df], ignore_index=True)

    df_dynamic = pd.DataFrame({
        'IPW_ATE_dynamic': np.mean(IPW_heterogeneous_ATE_Error, axis=0),
        'Doubly_Robust_ATE_dynamic': np.mean(Doubly_Robust_heterogeneous_ATE_Error, axis=0),
        'Kernel_ATE_dynamic': np.mean(Kernel_heterogeneous_ATE_Error, axis=0),
        'Kernel_functional_ATE_dynamic': np.mean(Kernel_functional_heterogeneous_ATE_Error, axis=0),
        'Kernel_SRVF_ATE_dynamic': np.mean(Kernel_SRVF_heterogeneous_ATE_Error, axis=0)
    })
    dynamic_dfs[n_samples] = df_dynamic

# Melt the combined DataFrame for seaborn boxplot
melted_df = combined_df.melt(id_vars='Sample_Size', var_name='Method', value_name='Error')

# Define a dictionary mapping for cleaner looking names on plots
rename_dict = {
    'IPW_ATE_dynamic': 'IPW ATE',
    'Doubly_Robust_ATE_dynamic': 'Doubly Robust ATE',
    'Kernel_ATE_dynamic': 'Kernel ATE',
    'Kernel_functional_ATE_dynamic': 'Operator Kernel ATE',
    'Kernel_SRVF_ATE_dynamic': 'SRVF Operator Kernel ATE'
}
rename_dict_error = {
    'IPW_ATE_Error': 'IPW ATE',
    'Doubly_Robust_ATE_Error': 'Doubly Robust ATE',
    'Kernel_ATE_Error': 'Kernel ATE',
    'Kernel_functional_ATE_Error': 'Operator Kernel ATE',
    'Kernel_SRVF_ATE_Error': 'SRVF Operator Kernel ATE'
}

# Rename method labels in melted DataFrame
melted_df['Method'] = melted_df['Method'].replace(rename_dict_error)

# Rename columns in dynamic dataframes
dynamic_dfs_renamed = {}
for sample_size, df_dyn in dynamic_dfs.items():
    renamed_df = df_dyn.rename(columns=rename_dict)
    dynamic_dfs_renamed[sample_size] = renamed_df

# Finally, plot
plot_combined_binary_treatment(melted_df, dynamic_dfs_renamed)

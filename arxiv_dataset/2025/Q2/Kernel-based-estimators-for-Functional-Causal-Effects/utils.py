# -*- coding: utf-8 -*-
"""
Utility functions for kernel-based causal estimators. If you use this code or any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.
@author: Yordan P. Raykov, Hengrui Luo 

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist
import scipy.stats as st
from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator

def compute_median_interpoint_distance(X):
    """
    Computes the median pairwise Euclidean distance for setting kernel_sigma.

    Parameters:
    - X: Input data of shape (n_samples, n_features)

    Returns:
    - Median of pairwise distances to be used as kernel_sigma.
    """
    pairwise_dists = pdist(X, metric='euclidean')  # Compute pairwise distances
    median_dist = np.median(pairwise_dists)  # Extract median
    return median_dist

def plot_combined_continious_treatment(melted_df, dynamic_dfs, save_plots=False, boxplot_params=None, lineplot_params=None):
    if boxplot_params is None:
        boxplot_params = {}
    if lineplot_params is None:
        lineplot_params = {}

    unique_sample_sizes = sorted(melted_df['Sample_Size'].unique())

    # --- Boxplot Subplots ---
    num_scenarios = len(unique_sample_sizes)
    fig = plt.figure(figsize=(6 * num_scenarios, 6))
    gs = fig.add_gridspec(1, num_scenarios + 1, width_ratios=[6] * num_scenarios + [2.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_scenarios)]

    for i, sample_size in enumerate(unique_sample_sizes):
        scenario_df = melted_df[melted_df['Sample_Size'] == sample_size]
        sns.boxplot(data=scenario_df, x='Sample_Size', y='Error', hue='Method', ax=axes[i], **boxplot_params)
        axes[i].set_title(f'Box Plot of ATE Errors (n={sample_size})', fontsize=18)
        axes[i].set_ylabel('Error', fontsize=16)
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].legend_.remove()

    # Place the legend on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Method', loc='center right', fontsize=14, title_fontsize=14)

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    if save_plots:
        plt.savefig('box_plots_combined_ate_errors.png', bbox_inches='tight')
    plt.show()

    # --- Dynamic Error Subplots ---
    num_dynamic = len(dynamic_dfs)
    fig = plt.figure(figsize=(6 * num_dynamic, 6))
    gs = fig.add_gridspec(1, num_dynamic + 1, width_ratios=[6] * num_dynamic + [2.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_dynamic)]

    for i, (sample_size, df_dynamic) in enumerate(dynamic_dfs.items()):
        sns.lineplot(data=df_dynamic, **lineplot_params, ax=axes[i])
        axes[i].set_ylabel('Estimation Error', fontsize=16)
        axes[i].set_xlabel('Time grid $C([0,1])$', fontsize=16)
        axes[i].grid()
        axes[i].legend_.remove()
        
        time_grid = np.linspace(0, 48, 48)
        ticks_original = axes[i].get_xticks()
        ticks_normalized = (ticks_original - time_grid.min()) / (time_grid.max() - time_grid.min())
        
        axes[i].set_xticks(ticks_original)
        axes[i].set_xticklabels(np.round(ticks_normalized, 2), fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].set_xlim(0, 47)
        axes[i].set_title(f'Dynamic Error (n={sample_size})', fontsize=18)

    # Place the legend on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Method', loc='center right', fontsize=14, title_fontsize=14)

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    if save_plots:
        plt.savefig('line_plots_dynamic_errors.png', bbox_inches='tight')
    plt.show()
    
def plot_combined_binary_treatment(
    melted_df,
    dynamic_dfs,
    save_plots=False,
    boxplot_params=None,
    lineplot_params=None
):
    """
    Plots boxplots of ATE errors for different sample sizes (subplots) 
    and dynamic error line plots for each sample size (subplots).
    
    Parameters
    ----------
    melted_df : pd.DataFrame
        A DataFrame containing columns ['Sample_Size', 'Method', 'Error'],
        typically created by melting a wider DataFrame of ATE errors.
    dynamic_dfs : dict of {int : pd.DataFrame}
        A dictionary mapping `sample_size -> DataFrame`, where each DataFrame
        has time-index columns of dynamic errors by method.
    save_plots : bool
        If True, saves the figures as PNGs.
    boxplot_params : dict
        Optional keyword arguments to pass directly to the boxplot call.
    lineplot_params : dict
        Optional keyword arguments to pass directly to the lineplot call.
    """
    if boxplot_params is None:
        boxplot_params = {}
    if lineplot_params is None:
        lineplot_params = {}

    import seaborn as sns  # Usually local import for plotting routines

    # Extract the distinct sample sizes present in the data
    unique_sample_sizes = sorted(melted_df['Sample_Size'].unique())

    # --- 1) Boxplot Subplots ---
    num_scenarios = len(unique_sample_sizes)
    fig, axes = plt.subplots(1, num_scenarios, figsize=(6 * num_scenarios, 6), sharey=True)

    # If only one scenario, make axes iterable
    if num_scenarios == 1:
        axes = [axes]

    for i, sample_size in enumerate(unique_sample_sizes):
        scenario_df = melted_df[melted_df['Sample_Size'] == sample_size]

        sns.boxplot(
            data=scenario_df,
            x='Sample_Size',
            y='Error',
            hue='Method',
            ax=axes[i],
            **boxplot_params
        )
        axes[i].set_title(f'Box Plot of ATE Errors (n={sample_size})', fontsize=18)
        axes[i].set_ylabel('Error', fontsize=16)

        # Remove x-axis labels (since they are just repeated sample_size)
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')

        # Increase font sizes
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].legend(fontsize=14, title_fontsize=14)

    plt.tight_layout()
    if save_plots:
        plt.savefig('box_plots_combined_ate_errors.png')
    plt.show()

    # --- 2) Dynamic Error Subplots ---
    fig, axes = plt.subplots(1, len(dynamic_dfs), figsize=(6 * len(dynamic_dfs), 6), sharey=True)
    # If there's only one scenario, make axes iterable
    if len(dynamic_dfs) == 1:
        axes = [axes]

    for i, (sample_size, df_dynamic) in enumerate(dynamic_dfs.items()):
        sns.lineplot(data=df_dynamic, ax=axes[i], **lineplot_params)

        axes[i].set_ylabel('Estimation Error', fontsize=16)
        axes[i].set_xlabel('Time grid $C([0,1])$', fontsize=16)
        axes[i].legend(title='Method', loc='upper right', fontsize=14, title_fontsize=14)
        axes[i].grid()

        # Normalize tick positions to [0, 1] for a consistent look
        n_timepoints = df_dynamic.shape[0]
        time_grid = np.linspace(0, n_timepoints, n_timepoints)

        # Get the default x-ticks as integers
        ticks_original = axes[i].get_xticks()
        # Convert them into [0,1] range
        ticks_normalized = (ticks_original - time_grid.min()) / (time_grid.max() - time_grid.min())

        axes[i].set_xticks(ticks_original)
        axes[i].set_xticklabels(np.round(ticks_normalized, 2), fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

        # Because time_grid goes from 0 to n_timepoints, the upper limit is n_timepoints-1:
        axes[i].set_xlim(0, n_timepoints - 1)
        axes[i].set_title(f'Dynamic Error (n={sample_size})', fontsize=18)

    plt.tight_layout()
    if save_plots:
        plt.savefig('line_plots_dynamic_errors.png')
    plt.show()
def generate_arc_data(n_timepoints, v, peak_perturbation=0.1):
    """
    Generates a half-circle arc function that starts and ends at 0,
    with amplitude controlled by v and a slightly perturbed peak position.

    Parameters:
    - n_timepoints (int): Number of time points.
    - v (float): Amplitude scaling factor (max height of the arc).
    - peak_perturbation (float): Fraction of the arc's domain to randomly shift peak.

    Returns:
    - y (ndarray): Arc values of shape (n_timepoints, 1).
    """
    t = np.linspace(0, 1, n_timepoints)  # Normalized time from 0 to 1

    # Random peak shift (perturbs the midpoint slightly)
    peak_shift = np.random.uniform(-peak_perturbation, peak_perturbation)
    t_shifted = t + peak_shift  # Shift time
    t_shifted = np.clip(t_shifted, 0, 1)  # Keep within [0,1]

    # Map time to [-1, 1] to simulate half-circle
    t_mapped = 2 * t_shifted - 1

    # Compute the arc using the half-circle equation: y = sqrt(1 - x^2)
    y = v * np.sqrt(np.clip(1 - t_mapped**2, 0, 1))  # Avoid sqrt of negative numbers

    return y.reshape(-1, 1)  # Ensure it's column-shaped

# Gaussian (RBF) kernel function
def rbf_kernel(x, xi, bandwidth):
    return np.exp(-np.linalg.norm(x - xi, axis=1) ** 2 / (2 * bandwidth ** 2))

# Nadaraya-Watson regression function
def nadaraya_watson(X_train, y_train, X_test, bandwidth):
    y_pred = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        weights = rbf_kernel(x, X_train, bandwidth)
        y_pred[i] = np.sum(weights * y_train) / np.sum(weights)
    return y_pred

# Function to evaluate the model given a set of hyperparameters
def evaluate_kernel_causal_effects(X_train, V_train, D_train, Y_train, X_test, V_test, D_test,
    theta1, kernel__lambd1, kernel__lambd2, kernel__lambd3, kernel_sigma,
    apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False):
    """
    Evaluate the model using kernel ridge regression causal estimation.

    Parameters:
    - X_train, V_train, D_train, Y_train: Training data.
    - X_test, V_test, D_test: Test data.
    - theta1: Ground truth ATE values.
    - kernel__lambd1, kernel__lambd2, kernel__lambd3: Regularization parameters.
    - kernel_sigma: Bandwidth parameter for the RBF kernel.
    - apply_srfv_Y: Whether to apply SRVF registration to Y_train.
    - srfv_Y_groups: Indices of groups for which SRVF registration should be applied to Y_train.
    - apply_srfv_X: Whether to apply SRVF registration to V_train.

    Returns:
    - Mean squared error (MSE) between true ATE and predicted ATE.
    """
    # Initialize the model with given hyperparameters and SRVF settings
    model = KernelRidgeRegressionCausalEstimator(
        lambd1=kernel__lambd1,
        lambd2=kernel__lambd2,
        lambd3=kernel__lambd3,
        kernel_sigma=kernel_sigma,
        use_operator_valued_kernel=False,  # Scalar kernel
        apply_srfv_Y=apply_srfv_Y,
        srfv_Y_groups=srfv_Y_groups,
        apply_srfv_X=apply_srfv_X
    )

    # Fit the model on training data
    theta_ate_vector_per_t, theta_cate_vector_per_t = model.fit(X_train, V_train, D_train, Y_train)

    # Predict on test data
    theta_ate_vector_pred, theta_cate_vector_pred = model.predict(X_test, V_test, D_test)

    # Calculate and return the mean squared error
    return mean_squared_error(theta1.flatten(), theta_ate_vector_pred.flatten())


def evaluate_func_kernel_causal_effects(X_train, V_train, D_train, Y_train, X_test, V_test, D_test, theta1, kernel__lambd1, kernel__lambd2, kernel__lambd3, kernel_sigma, apply_srfv_Y=False, srfv_Y_groups=None, apply_srfv_X=False, srfv_X_groups=None):
    """
    Evaluate the functional model using kernel ridge regression causal estimation.

    Parameters:
    - X_train, V_train, D_train, Y_train: Training data.
    - X_test, V_test, D_test: Test data.
    - theta1: Ground truth ATE values.
    - kernel__lambd1, kernel__lambd2, kernel__lambd3: Regularization parameters.
    - kernel_sigma: Bandwidth parameter for the RBF kernel.
    - apply_srfv_Y: Whether to apply SRVF registration to Y_train.
    - srfv_Y_groups: Indices of groups for which SRVF registration should be applied.
    - apply_srfv_X: Whether to apply SRVF registration to V_train.


    Returns:
    - Mean squared error (MSE) between true ATE and predicted ATE.
    """
    # Initialize the model with given hyperparameters and SRVF registration settings
    model = KernelRidgeRegressionCausalEstimator(
        lambd1=kernel__lambd1,
        lambd2=kernel__lambd2,
        lambd3=kernel__lambd3,
        kernel_sigma=kernel_sigma,
        use_operator_valued_kernel=True,
        apply_srfv_Y=apply_srfv_Y,
        srfv_Y_groups=srfv_Y_groups,
        apply_srfv_X=apply_srfv_X,
        srfv_X_groups=srfv_X_groups
    )

    # Fit the model on training data
    theta_ate_vector_per_t, theta_cate_vector_per_t = model.fit(X_train, V_train, D_train, Y_train)

    # Predict on test data
    theta_ate_vector_pred, theta_cate_vector_pred = model.predict(X_test, V_test, D_test)

    # Calculate and return the mean squared error
    return mean_squared_error(theta1.flatten(), theta_ate_vector_pred.flatten())

def perturb_array(x, displacement_range = 0.5):
    n_timepoints = len(x)
    perturbations = np.random.uniform(-displacement_range, displacement_range, size=n_timepoints)
    perturbed_x = x + perturbations
    return perturbed_x

def smooth_peak_curve(length, peak_location, peak_width=3):
    x = np.arange(length)
    curve = np.exp(-0.5 * ((x - peak_location) / peak_width)**2)  # Adjust the standard deviation for the smoothness
    return curve


# def generate_synthetic_curves(n_samples, 
#                               n_timepoints, 
#                               treatment='discrete', 
#                               noise_level=0.1,
#                               outcome_type='peak'):
#     """
#     Generates data for comparing estimation error between methods 
#     for causal function estimation with functional outputs.

#     - If scenario='peak' (default), uses your old logic (smooth_peak_curve)
#       for beta[0], beta[1], beta[2].
#     - If scenario='monotonic', we replace the beta curves with a monotonic
#       sequence or something that ensures mostly increasing shape.

#     The rest of the code remains essentially the same.
#     """
#     Y, D, V, X = [], [], [], []
#     theta_1_true_list = []

#     for _ in range(n_samples):
#         # ----------------------------------------------------------
#         # 1) Generate "beta" depending on scenario
#         # ----------------------------------------------------------
#         beta = np.zeros((3, n_timepoints))
#         perturb_step = np.random.uniform(-3, 3, size=1)

        
#         # Exactly your original approach for peak shapes
#         beta[0, :] = smooth_peak_curve(
#             n_timepoints, 
#             peak_location=6*(n_timepoints/24) + perturb_step, 
#             peak_width=2.5
#         )
#         beta[1, :] = smooth_peak_curve(
#             n_timepoints, 
#             peak_location=14*(n_timepoints/24) + perturb_step, 
#             peak_width=2.5
#         )
#         beta[2, :] = smooth_peak_curve(
#             n_timepoints, 
#             peak_location=20*(n_timepoints/24) + perturb_step, 
#             peak_width=2.5
#         )
            
        
#         # ----------------------------------------------------------
#         # 2) Generate other parameters as you already do
#         # ----------------------------------------------------------
#         epsilon = np.random.uniform(0.75, 1.25, 4)
#         v = epsilon[0]

#         x = np.zeros((3, 1))
#         x[0] = 1 + 2*v + epsilon[1]
#         x[1] = 1 + 2*v + epsilon[2]
#         x[2] = (v - 0.1)**2 + epsilon[3]

#         # ----------------------------------------------------------
#         # 3) Compute the "true treatment effect" exactly as before
#         # ----------------------------------------------------------
#         theta_1_true_sample = (
#             beta[0, :] * (1 + 2*v) +
#             beta[1, :] * (1 + 2*v) +
#             beta[2, :] * ((v - 0.1)**2)
#         )
#         if outcome_type == 'peak':
#             theta_1_true_list.append(theta_1_true_sample)
#         elif outcome_type == 'monotonic':
#             # Example: generate each beta[i,:] to be "mostly increasing".
#             # One simple way: a random slope plus small noise, then 
#             # ensure it doesn't decrease too much.
#             # You can choose any monotonic construction that suits you.
            
#             # Enforce monotonicity on theta_1_true
#             theta_1_true_sample = np.cumsum(np.abs(theta_1_true_sample))
#             theta_1_true_sample = theta_1_true_sample - theta_1_true_sample[0]  # Normalize to start from 0
#             theta_1_true_list.append(theta_1_true_sample)
#         else:
#             raise ValueError("outcome_type must be either 'peak' or 'monotonic'")
        

#         # ----------------------------------------------------------
#         # 4) Treatment assignment (same as before) & outcome generation
#         # ----------------------------------------------------------
#         if treatment == 'discrete':
#             # Random assignment via logistic function
#             d = np.random.binomial(
#                 n=1, 
#                 p=1 / (1 + np.exp(((v + np.sum(x))/2)/(1 - (v + np.sum(x))/2)))
#             )
#             D.append(d)
#             if d == 0:
#                 y = np.zeros((n_timepoints,))
#             else:
#                 # Add noise to the "true" effect
#                 y = theta_1_true_sample + np.random.normal(0, noise_level, n_timepoints)
#         else:
#             d = 0  # placeholder for non-discrete scenario
#             D.append(d)
#             y = np.zeros((n_timepoints,))  # placeholder

#         Y.append(y)
#         V.append(v)
#         X.append(x.flatten())

#     # Convert to numpy arrays
#     Y = np.array(Y)
#     D = np.array(D).reshape(n_samples, 1)
#     V = np.array(V).reshape(n_samples, 1)
#     X = np.array(X).reshape(n_samples, 3)
#     theta_1_true = np.array(theta_1_true_list)

#     return Y, D, V, X, theta_1_true


def generate_synthetic_curves(n_samples, 
                              n_timepoints, 
                              treatment='discrete', 
                              noise_level=0.1,
                              outcome_type='peak'):
    """
    A unified generator that reproduces:
      1) The original 'discrete' approach with different types of outcomes:
        - If scenario='peak' (default), uses your old logic (smooth_peak_curve) 
        for beta[0], beta[1], beta[2]. 
        - If scenario='monotonic', we replace the beta curves with a monotonic 
        sequence or something that ensures mostly increasing shape.
      2) The 'continuous' approach:
         - X is arc-based of length n_timepoints
         - Beta assignment
         - Y = arc + sum-of-peak-curves times d + noise

    Returns:
      Y: shape (n_samples, n_timepoints)
      D: shape (n_samples, 1)
      V: shape (n_samples, 1)
      X: list of length n_samples, each either shape (3,) or (n_timepoints,)
         (or we make it an object array if you want uniform type).
      theta_1_true: shape (n_samples, n_timepoints)
    """

    Y_list = []
    D_list = []
    V_list = []
    X_list = []
    theta_1_true_list = []

    for _ in range(n_samples):

        if treatment == 'discrete':
            # -------------------------------------------------
            # 1) Generate V, X of shape (3,)
            # -------------------------------------------------
            epsilon = np.random.uniform(0.75, 1.25, 4)
            v = epsilon[0]

            x_vec = np.zeros((3, 1))
            x_vec[0] = 1 + 2*v + epsilon[1]
            x_vec[1] = 1 + 2*v + epsilon[2]
            x_vec[2] = (v - 0.1)**2 + epsilon[3]

            # -------------------------------------------------
            # 2) Treatment assignment (logistic)
            # -------------------------------------------------
            p = 1 / (1 + np.exp(((v + np.sum(x_vec)) / 2) / (1 - (v + np.sum(x_vec)) / 2)))
            d = np.random.binomial(n=1, p=p)

            # -------------------------------------------------
            # 3) Generate peak-based betas
            # -------------------------------------------------
            beta = np.zeros((3, n_timepoints))
            perturb_step = np.random.uniform(-3, 3, size=1)
            # Three peak locations as in your code
            locs = [6, 14, 20]
            for i, peak_loc in enumerate(locs):
                beta[i, :] = smooth_peak_curve(
                    n_timepoints,
                    peak_location=peak_loc*(n_timepoints/24) + perturb_step,
                    peak_width=2.5
                )

            # -------------------------------------------------
            # 4) Compute true effect curve (same formula)
            # -------------------------------------------------
            theta_1_true_sample = (
                beta[0, :] * (1 + 2*v) +
                beta[1, :] * (1 + 2*v) +
                beta[2, :] * ((v - 0.1)**2)
            )
            if outcome_type == 'peak':
                theta_1_true_list.append(theta_1_true_sample)
            elif outcome_type == 'monotonic':
                # Example: generate each beta[i,:] to be "mostly increasing".
                # One simple way: a random slope plus small noise, then 
                # ensure it doesn't decrease too much.
                # You can choose any monotonic construction that suits you.
            
                # Enforce monotonicity on theta_1_true
                theta_1_true_sample = np.cumsum(np.abs(theta_1_true_sample))
                theta_1_true_sample = theta_1_true_sample - theta_1_true_sample[0]  # Normalize to start from 0
                theta_1_true_list.append(theta_1_true_sample)
            else:
                raise ValueError("outcome_type must be either 'peak' or 'monotonic'")
            # -------------------------------------------------
            # 5) Generate outcome
            # -------------------------------------------------
            if d == 0:
                y = np.zeros(n_timepoints)
            else:
                y = theta_1_true_sample + np.random.normal(0, noise_level, n_timepoints)

            # For storing the "true effect" * d
            # (so if d=0, effect is 0 in that sample)
          #  theta_1_true_sample *= float(d)

            # Save everything
            Y_list.append(y)
            D_list.append(d)
            V_list.append(v)
            X_list.append(x_vec.flatten())  # shape (3,)

        elif treatment == 'continuous':
            # -------------------------------------------------
            # 1) Generate V, X of shape (n_timepoints,)
            #    X = arc_data + small noise
            # -------------------------------------------------
            epsilon = np.random.uniform(0.15, 0.75, 1)
            v = epsilon[0]

            noise = np.random.uniform(0.1, 0.15, size=n_timepoints)
            x_arc = generate_arc_data(n_timepoints, v).flatten()  # shape (n_timepoints,)
            x_arc = x_arc + noise  # still shape (n_timepoints,)

            # -------------------------------------------------
            # 2) Treatment assignment (Beta) 
            # -------------------------------------------------
            alpha_par = 1 + np.abs((v + np.sum(x_arc)) / 10)
            beta_par = 1 + np.abs((v + np.sum(x_arc)) / 10)
            d = np.random.beta(alpha_par, beta_par)  # continuous in [0, 1]

            # -------------------------------------------------
            # 3) Generate peak-based betas
            # -------------------------------------------------
            beta = np.zeros((3, n_timepoints))
            perturb_step = np.random.uniform(-3, 3, size=1)
            locs = [6, 14, 20]
            for i, peak_loc in enumerate(locs):
                beta[i, :] = smooth_peak_curve(
                    n_timepoints,
                    peak_location=peak_loc*(n_timepoints/24) + perturb_step,
                    peak_width=2.5
                )

            # -------------------------------------------------
            # 4) True effect curve
            #    Per your second code: sum of the 3 betas + arc_data
            # -------------------------------------------------
            arc_data_2 = generate_arc_data(n_timepoints, v).flatten()
            theta_1_true_sample = (beta[0, :] + beta[1, :] + beta[2, :] + arc_data_2)

            # -------------------------------------------------
            # 5) Generate outcome = d * effect + noise
            # -------------------------------------------------
            y = d * theta_1_true_sample + np.random.normal(0, noise_level, n_timepoints)

            # For storing the "true effect" * d
           # theta_1_true_sample *= float(d)

            # Save everything
            Y_list.append(y)
            D_list.append(d)
            V_list.append(v)
            X_list.append(x_arc)  # shape (n_timepoints,)
            theta_1_true_list.append(theta_1_true_sample)

        else:
            raise ValueError("treatment must be 'discrete' or 'continuous'.")

    # -----------------------------------------
    # Convert to arrays
    # -----------------------------------------
    Y = np.array(Y_list)  # shape (n_samples, n_timepoints)
    D = np.array(D_list).reshape(n_samples, 1)
    V = np.array(V_list).reshape(n_samples, 1)

    if treatment == 'discrete':    
        X = np.array(X_list).reshape(n_samples, 3)  
    elif treatment == 'continuous':
        X = np.array(X_list).reshape(n_samples, n_timepoints)
            

    theta_1_true = np.array(theta_1_true_list)  # (n_samples, n_timepoints)

    return Y, D, V, X, theta_1_true



# Compute the Fisher-Rao distance between the two SRVFs
def fisher_rao_distance(q1, q2, t):
    # Ensure q1 and q2 have the same length as t
    if len(q1) != len(t) or len(q2) != len(t):
        raise ValueError("The length of q1, q2, and t must be the same.")

    # Compute the Fisher-Rao distance
    integrand = (q1 - q2) ** 2
    distance = np.sqrt(np.trapz(integrand, t))

    return distance


def reshape_and_split_data(Y, D, V, X, theta_1_true, frac_train, random_state=42):
    """
    Splits the data into training and test sets, ensuring alignment across all arrays.

    Parameters:
    - Y: Outcome variable data (n_samples, n_timepoints).
    - D: Treatment indicator data (n_samples, 1).
    - V: Covariate data (n_samples, 1).
    - X: Covariate data (n_samples, n_features).
    - theta_1_true: True treatment effect curves for the data (n_samples, n_timepoints).
    - frac_train: Fraction of data to use for training.
    - random_state: Random seed for reproducibility.

    Returns:
    - Y_train, D_train, X_train, V_train, theta_1_true_train: Training data.
    - Y_test, D_test, X_test, V_test, theta_true_test: Test data.
    """
    # Ensure that D, V, and Y have correct shapes
    D = D.reshape(-1, 1)
    V = V.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if theta_1_true.ndim == 1:
        theta_1_true = theta_1_true.reshape(-1, 1)

    # Split the data using train_test_split, maintaining alignment
    Y_train, Y_test, D_train, D_test, V_train, V_test, X_train, X_test, theta_1_true_train, theta_true_test = train_test_split(
        Y, D, V, X, theta_1_true, test_size=1 - frac_train, random_state=random_state
    )

    return Y_train, D_train, X_train, V_train, theta_1_true_train, Y_test, D_test, X_test, V_test, theta_true_test

def cross_validate_bandwidth(X_train, D_train, bandwidths, k=5):
    """
    Performs k-fold cross-validation to find the optimal bandwidth for Nadaraya-Watson regression.

    Parameters:
    X_train (ndarray): Training feature data.
    D_train (ndarray): Training target data.
    bandwidths (list of float): List of bandwidths to evaluate.
    k (int): Number of cross-validation folds.

    Returns:
    optimal_bandwidth (float): The bandwidth with the lowest average validation error.
    """

    from sklearn.metrics import mean_squared_error
    import numpy as np

    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    avg_errors = []

    for bandwidth in bandwidths:
        errors = []
        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            D_fold_train, D_fold_val = D_train[train_index], D_train[val_index]
            D_fold_pred = nadaraya_watson(X_fold_train, D_fold_train, X_fold_val, bandwidth)

            # Check for NaN and skip fold if found
            if np.isnan(D_fold_pred).any() or np.isnan(D_fold_val).any():
                print("Skipping fold due to NaNs")
                continue

            error = mean_squared_error(D_fold_val, D_fold_pred)
            errors.append(error)

        if errors:  # Avoid empty list if folds are skipped
            avg_error = np.mean(errors)
        else:
            avg_error = np.inf  # Set high error if all folds fail
        avg_errors.append(avg_error)
        print(f"Bandwidth: {bandwidth}, Average Validation Error: {avg_error}")

    optimal_bandwidth = bandwidths[np.argmin(avg_errors)]
    return optimal_bandwidth

def test_zero_effect(Delta_hat, S, n, alpha=0.05):
    """
    Test H0: ||Delta||^2 = 0 vs. H1: ||Delta||^2 > 0 using the asymptotic
    distribution of sqrt(n)*||Delta_hat||^2 => ||Z||^2 for Z ~ N(0, K).
    
    If we assume K=I_d, then ||Z||^2 ~ chi^2_{T}. More generally,
    if we diagonalize S, we get a generalized chi^2 distribution. 
    
    For a quick demonstration we can do the naive approach of
    using a standard chi^2_{T} with T = dimension of Delta,
    i.e. ignoring the correlation.  Then:
        test_stat = n * ||Delta_hat||^2   (no sqrt needed if you read Theorem carefully).
    or you might do sqrt(n)*||Delta_hat|| if you prefer matching Theorem exactly.

    For a more precise approach, one would do an eigen-decomposition of S
    => a generalized chi^2 quantile. For simplicity, here's a naive version:
    """
    T = Delta_hat.shape[0]
    norm_sq_est = np.sum(Delta_hat**2)
    
    # Simple test statistic (one version):
    #   sqrt(n)*||Delta_hat||^2   ~ distribution of ||Z||^2
    # We'll do "n* norm_sq_est" for convenience
    test_stat = n * norm_sq_est
    
    # Compare to chi^2_{T} quantile
    chi2_crit = st.chi2.ppf(1 - alpha, df=T)
    
    p_value = 1 - st.chi2.cdf(test_stat, df=T)

    reject = (test_stat > chi2_crit)
    
    return test_stat, chi2_crit, p_value, reject

def delta_method_ci(Delta_hat, S, n, alpha=0.05):
    """
    Build a normal-approx confidence interval for ||Delta||^2 using the delta method:
        Var(||X||^2) ~ 4 * (X^T Cov(X) X) / n
    where X=Delta_hat in R^T, Cov(X) ~ S/n, etc.
    
    - Delta_hat : shape (T,) for the average effect
    - S : shape (T,T) the sample covariance of the single-subject effect vectors
    - n : number of test samples
    - alpha : significance level for the (1-alpha) CI

    Returns: (L, U) = lower and upper endpoints for the CI on ||Delta||^2.
    """
    # 1) Norm-squared of the average effect
    norm_sq_est = np.sum(Delta_hat**2)  # ||Delta_hat||^2

    # 2) Delta method approximate var( ||Delta||^2 ) = 4 * Delta_hat^T S Delta_hat / n
    #    because the gradient of g(x)=||x||^2 is 2x, so Var(g(X)) ~ grad(g)^T Cov(X) grad(g).
    #    Cov(X) ~ S/n, so overall factor is 4 (Delta_hat^T S Delta_hat) / n
    grad = 2.0 * Delta_hat.reshape(-1,1)  # shape (T,1)
    var_est = (grad.T @ S @ grad).item() / n  # ~ (4 * Delta_hat^T S Delta_hat)/n if S is Cov(X).
    
    # But note: S is the sample covariance of Delta_i (not the mean!), so
    # Cov(mean(Delta_i)) = S / n. That means:
    # var(||Delta_hat||^2) ~ 4*(Delta_hat^T [S/n] Delta_hat) = (4/n)(Delta_hat^T S Delta_hat).
    # Let's do it explicitly:
    var_approx = 4.0 * (Delta_hat @ (S @ Delta_hat)) / (n)
    
    std_err = np.sqrt(var_approx)
    
    # 3) Normal quantile
    z_val = st.norm.ppf(1 - alpha/2)
    
    # 4) confidence interval
    lower = norm_sq_est - z_val * std_err
    upper = norm_sq_est + z_val * std_err
    
    # ensure non-negativity if you want:
    lower = max(lower, 0.0)

    return (lower, upper)

def estimate_covariance(Delta_hat_i):
    """
    Estimate the covariance matrix of the random vector Delta_i = Y_i(1) - Y_i(0).
    We'll do the usual sample covariance (biased or unbiased).
    Delta_hat_i: shape (n_test, T)
    returns a (T x T) sample covariance matrix
    """
    n_test, T = Delta_hat_i.shape
    # compute sample mean
    mean_vec = np.mean(Delta_hat_i, axis=0, keepdims=True)
    # center
    centered = Delta_hat_i - mean_vec
    # sample covariance
    # You could use np.cov(..., rowvar=False), but let's be explicit:
    S = (centered.T @ centered) / (n_test - 1)  # shape (T, T)
    return S

def compute_estimated_effects(predictions_cate):
    """
    Given predicted_cate of shape (n_test, 2, T),
    returns:
       Delta_hat_i : array of shape (n_test, T), the individual-level effect curves
       Delta_hat   : array of shape (T,), the average effect curve
    """
    # predictions_cate[i,0,:] = predicted Y(0) for subject i
    # predictions_cate[i,1,:] = predicted Y(1) for subject i
    
    # effect for each subject i
    Delta_hat_i = predictions_cate[:,1,:] - predictions_cate[:,0,:]  # shape (n_test, T)
    
    # average effect over subjects
    Delta_hat = np.mean(Delta_hat_i, axis=0)  # shape (T,)
    
    return Delta_hat_i, Delta_hat
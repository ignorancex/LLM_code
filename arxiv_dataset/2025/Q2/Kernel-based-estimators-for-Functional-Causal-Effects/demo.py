# -*- coding: utf-8 -*-
"""

Demo comparing operator-valued kernel with vs. without SRVF registration
on a single simulated dataset (n=50). If you use this code or any part of it in your research, please cite the following paper:
Raykov, Y.P., Luo, H., Strait, J.D. and KhudaBukhsh, W.R., 2025. Kernel-based estimators for functional causal effects. arXiv preprint arXiv:2503.05024.
@author: Yordan P. Raykov
"""

import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from utils import (
    generate_synthetic_curves,
    compute_median_interpoint_distance,
    evaluate_func_kernel_causal_effects,
)
from KernelRidgeRegressionCausalEstimator import KernelRidgeRegressionCausalEstimator

# ---------------------------------------------------------
# 1) Generate a single simulated dataset (n=50) 
#    (binary treatment, functional outcome)
# ---------------------------------------------------------
n_samples = 50
n_timepoints = 48
frac_train = 0.80

Y, D, V, X, theta_1_true = generate_synthetic_curves(n_samples, n_timepoints)  
# shape(Y) = (n_samples, n_timepoints)

# Flatten the treatment and V for easier splitting
D = D.ravel() 
V = V.ravel()

# Train/test split
X_train, X_test, Y_train, Y_test, D_train, D_test, V_train, V_test, theta_1_true_train, theta_1_true_test = \
    train_test_split(X, Y, D, V, theta_1_true, test_size=(1-frac_train), random_state=42)

# ---------------------------------------------------------
# 2) Compute kernel parameter (sigma) from training features
# ---------------------------------------------------------
kernel_sigma = compute_median_interpoint_distance(X_train)
print("Median-based kernel_sigma:", kernel_sigma)

# ---------------------------------------------------------
# 3) Hyperparameter search for operator-valued kernel 
#    WITHOUT registration
# ---------------------------------------------------------
lambd1_vals = np.logspace(-3, 3, 3)
lambd2_vals = np.logspace(-3, 3, 3)
lambd3_vals = np.logspace(-3, 3, 3)
hyperparameter_grid = list(product(lambd1_vals, lambd2_vals, lambd3_vals))

best_score_no_reg = float('inf')
best_params_no_reg = None

for lam1, lam2, lam3 in hyperparameter_grid:
    score = evaluate_func_kernel_causal_effects(
        X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train,
        X_test,  V_test.reshape(-1,1),  D_test.reshape(-1,1),  Y_test,
        lam1, lam2, lam3, kernel_sigma,
        apply_srfv_Y=False,  # <--- No SRVF registration
        srfv_Y_groups=None, 
        apply_srfv_X=False
    )
    if score < best_score_no_reg:
        best_score_no_reg = score
        best_params_no_reg = (lam1, lam2, lam3)

print(f"\n[No Registration] Best hyperparams: lambda1={best_params_no_reg[0]}, "
      f"lambda2={best_params_no_reg[1]}, lambda3={best_params_no_reg[2]}, "
      f"MSE={best_score_no_reg:.4f}")

# Fit final operator-valued estimator (no registration) on the entire training set
kernel_causal_estimator_no_reg = KernelRidgeRegressionCausalEstimator(
    lambd1=best_params_no_reg[0], 
    lambd2=best_params_no_reg[1], 
    lambd3=best_params_no_reg[2],
    kernel_sigma=kernel_sigma, 
    treatment='discrete',
    use_operator_valued_kernel=True,
    apply_srfv_Y=False,
    srfv_Y_groups=None,
    apply_srfv_X=False
)

kernel_causal_estimator_no_reg.fit(X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train)
# For the test set, we get predicted ATE for each subject (i.e., E[Y(1)-Y(0)|X_test])
predictions_ate_no_reg, predictions_cate_no_reg = kernel_causal_estimator_no_reg.predict(
    X_test, V_test.reshape(-1,1), D_test.reshape(-1,1)
)

# Evaluate average integrated error
# (comparing predicted Y(1) - Y(0) vs true theta_1)
test_truth = theta_1_true_test  # shape: (n_test, n_timepoints)
est_diff_no_reg = predictions_cate_no_reg[:,1,:] - predictions_cate_no_reg[:,0,:]
mse_no_reg = np.mean((est_diff_no_reg - test_truth)**2)
print(f"Operator-valued Kernel WITHOUT SRVF registration, MSE on test set: {mse_no_reg:.4f}")

# ---------------------------------------------------------
# 4) Hyperparameter search for operator-valued kernel 
#    WITH SRVF registration
# ---------------------------------------------------------
best_score_srvf = float('inf')
best_params_srvf = None

for lam1, lam2, lam3 in hyperparameter_grid:
    score = evaluate_func_kernel_causal_effects(
        X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train,
        X_test,  V_test.reshape(-1,1),  D_test.reshape(-1,1),  Y_test,
        lam1, lam2, lam3, kernel_sigma,
        apply_srfv_Y=True,   # <--- Turn on SRVF registration
        srfv_Y_groups=[1],  # (We have only one outcome group, so group=[1])
        apply_srfv_X=False
    )
    if score < best_score_srvf:
        best_score_srvf = score
        best_params_srvf = (lam1, lam2, lam3)

print(f"\n[With SRVF] Best hyperparams: lambda1={best_params_srvf[0]}, "
      f"lambda2={best_params_srvf[1]}, lambda3={best_params_srvf[2]}, "
      f"MSE={best_score_srvf:.4f}")

# Fit final operator-valued estimator (with registration) on the entire training set
kernel_causal_estimator_srvf = KernelRidgeRegressionCausalEstimator(
    lambd1=best_params_srvf[0], 
    lambd2=best_params_srvf[1], 
    lambd3=best_params_srvf[2],
    kernel_sigma=kernel_sigma, 
    treatment='discrete',
    use_operator_valued_kernel=True,
    apply_srfv_Y=True, 
    srfv_Y_groups=[1],
    apply_srfv_X=False
)

kernel_causal_estimator_srvf.fit(X_train, V_train.reshape(-1,1), D_train.reshape(-1,1), Y_train)
predictions_ate_srvf, predictions_cate_srvf = kernel_causal_estimator_srvf.predict(
    X_test, V_test.reshape(-1,1), D_test.reshape(-1,1)
)

# Evaluate average integrated error for the SRVF approach
est_diff_srvf = predictions_cate_srvf[:,1,:] - predictions_cate_srvf[:,0,:]
mse_srvf = np.mean((est_diff_srvf - test_truth)**2)
print(f"Operator-valued Kernel WITH SRVF registration, MSE on test set: {mse_srvf:.4f}")

# ---------------------------------------------------------
# 5) Confidence intervals and significance of the effects
# ---------------------------------------------------------
from utils import compute_estimated_effects, estimate_covariance, delta_method_ci, test_zero_effect

# 1) Compute estimated effect (subject-level and average)
Delta_hat_i, Delta_hat = compute_estimated_effects(predictions_cate_srvf)

# 2) Estimate covariance of the subject-level effect
S = estimate_covariance(Delta_hat_i)
n_test = Delta_hat_i.shape[0]

# 3) Build a delta-method CI for ||Delta||^2 (assuming the true norm is not zero):
ci_lower, ci_upper = delta_method_ci(Delta_hat, S, n_test, alpha=0.05)
print(f"Estimated ||Delta||^2 = {np.sum(Delta_hat**2):.4f}")
print(f"95% CI (non-zero effect assumption) = [{ci_lower:.4f}, {ci_upper:.4f}]")

# 4) (Optional) Test for zero effect vs. positive effect
test_stat, chi2_crit, p_value, reject = test_zero_effect(Delta_hat, S, n_test, alpha=0.05)
print(f"Test statistic (naive chi^2) = {test_stat:.4f}")
print(f"Chi^2 critical @ 95%        = {chi2_crit:.4f}")
print(f"p-value                      = {p_value:.4f}")
print(f"Reject H0 (zero effect)?     = {reject}")

# ---------------------
# Plotting
# ---------------------

n_test, T = predictions_ate_srvf.shape
time_grid = np.arange(T)

# 1) Mean ATE across subjects at each timepoint
mean_ate = np.mean(predictions_ate_srvf, axis=0)  # shape (T,)

# 2) Standard error (assuming independence across test subjects)
std_ate = np.std(predictions_ate_srvf, axis=0, ddof=1)  # sample stdev across i
sem_ate = std_ate / np.sqrt(n_test)                     # standard error

# 3) For a 95% confidence interval, use 1.96 as the normal critical value
z_val = 1.96
ci_lower = mean_ate - z_val * sem_ate
ci_upper = mean_ate + z_val * sem_ate

import matplotlib.pyplot as plt

plt.plot(time_grid, mean_ate, label="Mean SRVF ATE")
plt.fill_between(time_grid, ci_lower, ci_upper, alpha=0.2, label="95% CI")

plt.xlabel("Time index")
plt.ylabel("Predicted ATE")
plt.title("SRVF Operator‐Valued Kernel – ATE over Time")
plt.legend()
plt.show()


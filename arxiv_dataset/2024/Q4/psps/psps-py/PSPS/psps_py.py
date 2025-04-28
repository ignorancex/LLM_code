import numpy as np
import pandas as pd
from scipy.stats import norm

def psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma, alpha=0.05):
    """
    PoSt-Prediction Sumstats-based inference.

    Parameters:
    - est_lab_y: A K-dimensional array of point estimates using Y in labeled data.
    - est_lab_yhat: A K-dimensional array of point estimates using Yhat in labeled data.
    - est_unlab_yhat: A K-dimensional array of point estimates using Yhat in unlabeled data.
    - Sigma: A 3K x 3K variance-covariance matrix for the above three estimators (note: not the asymptotic variance).
    - alpha: Specifies the confidence level as 1 - alpha for confidence intervals.
    - table_name: The name or title for the output table.
    
    Returns:
    - A pandas DataFrame containing the summary table with columns for estimates, standard errors, lower confidence interval,
      upper confidence interval, and P-values, labeled with a table name.
    """
    K = len(np.array([est_lab_y]))
    Sigma_matrix = np.array(Sigma)
    v_est = Sigma_matrix[0:K, 0:K]
    r = Sigma_matrix[K:2*K, 0:K]
    v_eta_lab = Sigma_matrix[K:2*K, K:2*K]
    v_eta_unlab = Sigma_matrix[2*K:3*K, 2*K:3*K]
    V = v_eta_lab + v_eta_unlab
    omega_0 = np.linalg.solve(V, r)

    # Calculate results
    est = est_lab_y + np.dot(omega_0, (est_unlab_yhat - est_lab_yhat))
    standard_errors = np.sqrt(np.diag(v_est - np.dot(omega_0.T, r)))
    lower_ci = est - norm.ppf(1 - alpha / 2) * standard_errors
    upper_ci = est + norm.ppf(1 - alpha / 2) * standard_errors
    p_values = 2 * norm.sf(np.abs(est / standard_errors))

    output_table = pd.DataFrame({
        'Estimate': np.ravel(est), 
        'Std.Error': np.ravel(standard_errors), 
        'Lower.CI': np.ravel(lower_ci), 
        'Upper.CI': np.ravel(upper_ci), 
        'P.value': np.ravel(p_values)
    })
        
    output_table.attrs['Table Name'] = 'psps summary'
    return output_table

# Example usage:
# Define Sigma and estimation vectors appropriately before calling the function.
# results = psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
# print(results)
# print("Table Name:", results.attrs['Table Name'])

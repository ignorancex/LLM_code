import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pdb

def generate_covariance_matrix(n_variables, var_min=0.1, var_max=1.0, random_seed=42):
    """
    Generate a covariance matrix with a controllable range of variances
    :param n_variables: Number of variables
    :param var_min: Minimum variance
    :param var_max: Maximum variance
    :param random_seed: Random seed for reproducibility
    :return: Positive definite covariance matrix
    """
    np.random.seed(random_seed)
    # Generate diagonal variance values
    variances = np.random.uniform(var_min, var_max, n_variables)
    cov_matrix = np.diag(variances)  # Initialize as a diagonal matrix

    # Add off-diagonal correlations (ensure positive definiteness)
    random_off_diag = np.random.uniform(-0.1, 0.1, (n_variables, n_variables))  # Random values for off-diagonal
    random_off_diag = (random_off_diag + random_off_diag.T) / 2  # Ensure symmetry
    cov_matrix += random_off_diag

    # Correct to a positive definite matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals[eigvals < 1e-6] = 1e-6  # Adjust negative eigenvalues to small positive values to ensure positive definiteness
    cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_matrix

# Read CSV file
file_path = "encoded_kincade_same.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Indicator columns and independent variable columns
indicators = ['Q44','Q46']
othe_indicators=['Q14','Q15','Q16','Q17','Q18','Q19', 'Q25','Q26','Q27','Q28','Q29','Q30','Q34', 'Q35','Q36','Q37','Q38','Q39','Q42','Q43','Q41','Q45']
exclude=indicators+othe_indicators
variables = [col for col in df.columns if col not in exclude]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[variables])
X = pd.DataFrame(X_scaled, columns=variables)

# Initialize result storage
weights_summary = pd.DataFrame()

# Construct covariance matrix Omega
n_variables = len(variables)
# np.random.seed(42)  # Fixed random seed for reproducibility
# cov_matrix = np.random.rand(n_variables, n_variables)
# cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Ensure matrix is positive definite
# cov_matrix /= np.max(cov_matrix)  # Normalize to control the range of covariance
var_min = 0  # Minimum variance
var_max = 0.1  # Maximum variance
cov_matrix = generate_covariance_matrix(n_variables, var_min, var_max)

# Used to store fitting information for each indicator
indicator_results = {}

# Perform regression for each indicator, calculate weights, and introduce multidimensional error terms
for indicator in indicators:
    y = df[indicator]
    
    # Generate multidimensional normal distribution error term epsilon_T
    epsilon_T = np.random.multivariate_normal(mean=np.zeros(n_variables), cov=cov_matrix, size=len(y))
    y_with_error = y + epsilon_T[:, 0]  # Take the first column error to affect the dependent variable
    
    # Regression fitting using all variables
    model = LinearRegression()
    model.fit(X, y_with_error)
    coef = model.coef_
    baseline_score = model.score(X, y_with_error)  # Use R² as fitting degree evaluation metric
    
    # Save weight information
    weights_df = pd.DataFrame({'Variable': variables, 'Weight': np.abs(coef)})
    weights_df['Indicator'] = indicator
    weights_summary = pd.concat([weights_summary, weights_df])
    
    # Store the original results of each indicator for refitting with selected variables later
    indicator_results[indicator] = {
        'y_with_error': y_with_error,
        'baseline_score': baseline_score
    }

# Calculate total weights and sort
weights_summary = weights_summary.groupby('Variable', as_index=False)['Weight'].sum()
weights_summary = weights_summary.sort_values(by='Weight', ascending=False)

# Cumulative weights
weights_summary['Cumulative_Weight'] = weights_summary['Weight'].cumsum()
total_weight = weights_summary['Weight'].sum()
weights_summary['Cumulative_Percentage'] = weights_summary['Cumulative_Weight'] / total_weight

# Select variables that meet the conditions
selected_variables = weights_summary[weights_summary['Cumulative_Percentage'] <= 0.7]

print("Selected variables:")
print(selected_variables)

# Save selected variables
selected_variables.to_csv("selected_variables_with_cov_error.csv", index=False)

# Refit using selected variables and calculate fitting scores
X_selected = X[selected_variables['Variable']]

for indicator in indicators:
    y_with_error = indicator_results[indicator]['y_with_error']
    # Refit using selected variables
    model_sel = LinearRegression()
    model_sel.fit(X_selected, y_with_error)
    selected_score = model_sel.score(X_selected, y_with_error)
    
    print(f"Indicator {indicator} R² using all variables: {indicator_results[indicator]['baseline_score']:.4f}")
    print(f"Indicator {indicator} R² using selected variables: {selected_score:.4f}\n")

# Plot cumulative weight percentage curve
plt.figure()
plt.plot(range(len(weights_summary)), weights_summary['Cumulative_Percentage'], marker='o')
plt.xlabel('Variable Index (Sorted by Weight)')
plt.ylabel('Cumulative Percentage of Weights')
plt.title('Cumulative Weight Percentage of All Variables')
plt.grid(True)
plt.show()

# Plot weight decreasing trend
plt.figure()
plt.plot(range(len(weights_summary)), weights_summary['Weight'], marker='o')
plt.xlabel('Variable Index (Sorted by Weight)')
plt.ylabel('Weight')
plt.title('Weight Decreasing Trend of All Variables')
plt.grid(True)
plt.show()

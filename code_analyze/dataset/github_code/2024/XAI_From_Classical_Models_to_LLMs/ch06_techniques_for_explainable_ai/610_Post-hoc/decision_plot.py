import shap
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor

# Load the California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Subsample the data (e.g., 1000 observations)
subset_size = 1000
random_indices = np.random.choice(X.shape[0], subset_size, replace=False)
X_subset = X[random_indices]
shap_values_subset = shap_values[random_indices]

# Generate the decision plot with subsampled data
shap.decision_plot(explainer.expected_value, shap_values_subset, X_subset, feature_names=data.feature_names)

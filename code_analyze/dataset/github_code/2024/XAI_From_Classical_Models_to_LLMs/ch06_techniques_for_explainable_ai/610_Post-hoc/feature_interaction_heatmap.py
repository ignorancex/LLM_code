import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Compute SHAP interaction values
explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X)

# Extract the main SHAP values from the interaction matrix (diagonal elements only)
shap_values_main = np.array([shap_interaction_values[i][:, i] for i in range(X.shape[1])]).T

# Plot the Feature Interaction Heatmap
shap.summary_plot(shap_values_main, X, feature_names=feature_names, plot_type="bar")
plt.title("Feature Importance for California Housing Dataset")
plt.show()

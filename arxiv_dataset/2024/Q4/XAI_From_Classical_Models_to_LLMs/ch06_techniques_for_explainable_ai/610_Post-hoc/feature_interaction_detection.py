import shap
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Load the California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Train the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create SHAP explainer and compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Compute SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X)

# Plot the SHAP interaction values (summary plot)
shap.summary_plot(shap_interaction_values, X, feature_names=feature_names, plot_type="compact_dot")

# Visualize a specific feature pair interaction
shap.dependence_plot(("MedInc", "AveRooms"), shap_interaction_values, X, feature_names=feature_names, interaction_index="AveRooms")

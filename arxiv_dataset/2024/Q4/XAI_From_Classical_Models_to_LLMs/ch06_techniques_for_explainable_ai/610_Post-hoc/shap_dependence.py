import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Extract SHAP values for dependence plot
# Using `shap_values.values` to get the actual SHAP values as an array
shap.dependence_plot("AveRooms", shap_values.values, X, feature_names=feature_names)
plt.title("SHAP Dependence Plot for AveRooms")
plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("SHAP Value (Impact on Model Output)")
plt.show()

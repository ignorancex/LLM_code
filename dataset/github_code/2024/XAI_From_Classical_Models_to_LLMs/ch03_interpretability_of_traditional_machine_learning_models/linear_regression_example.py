import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example data: [Square Footage, Number of Bedrooms]
X = np.array([[1500, 3], [2000, 4], [2500, 4], [3000, 5]])
y = np.array([300000, 400000, 500000, 600000])

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Output the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict house prices using the trained model
y_pred = model.predict(X)
print("Predicted Prices:", y_pred)

# Plot the true vs predicted prices
plt.scatter(range(len(y)), y, color='blue', label='True Prices')
plt.scatter(range(len(y_pred)), y_pred, color='red', marker='x', label='Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('House Price ($)')
plt.title('True vs Predicted House Prices')
plt.legend()
plt.show()

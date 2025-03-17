import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Example data: [Monthly Charges, Tenure]
X = np.array([[30, 1], [40, 3], [50, 5], [60, 7]])
y = np.array([0, 0, 1, 1])  # 0 = No churn, 1 = Churn

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Output the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions and predict probabilities
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print("Predicted Labels:", y_pred)
print("Predicted Probabilities (Churn):", y_prob)

# Visualization of the predicted probabilities
plt.scatter(range(len(y)), y, color='blue', label='True Labels (0=No churn, 1=Churn)')
plt.plot(range(len(y_prob)), y_prob, color='red', marker='x', linestyle='--', label='Predicted Probabilities')
plt.xlabel('Sample Index')
plt.ylabel('Probability of Churn')
plt.title('Logistic Regression: Churn Prediction')
plt.legend()
plt.show()

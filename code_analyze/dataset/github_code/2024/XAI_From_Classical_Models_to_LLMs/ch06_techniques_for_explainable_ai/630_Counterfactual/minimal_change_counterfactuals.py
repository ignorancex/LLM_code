import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

# Create a synthetic dataset
X = np.array([[0.1, 0.5], [0.4, 0.8], [0.5, 0.3], [0.9, 0.6], [0.7, 0.9]])
y = np.array([0, 0, 0, 1, 1])

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X, y)

# Define the objective function for minimal change counterfactual
def objective_function(x_prime, x, model, target_class):
    distance = np.linalg.norm(x - x_prime)
    prediction = model.predict([x_prime])[0]
    classification_loss = 0 if prediction == target_class else 1
    return distance + 10 * classification_loss  # Penalize if prediction does not match target class

# Generate a minimal change counterfactual
def generate_minimal_counterfactual(model, x, target_class):
    x_prime = np.copy(x)
    result = minimize(
        objective_function,
        x_prime,
        args=(x, model, target_class),
        method='L-BFGS-B'
    )
    return result.x

# Example input and target class
x = np.array([0.3, 0.7])
target_class = 1  # Desired outcome different from the model's original prediction

# Generate the counterfactual
counterfactual = generate_minimal_counterfactual(model, x, target_class)

print("Original input:", x)
print("Minimal change counterfactual:", counterfactual)
print("Original prediction:", model.predict([x])[0])
print("Counterfactual prediction:", model.predict([counterfactual])[0])
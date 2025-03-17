import numpy as np
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize

# Synthetic dataset: Features are age, income, credit score, and debt-to-income ratio
X = np.array([[25, 40000, 650, 0.3], [45, 80000, 720, 0.2], [35, 60000, 690, 0.25], [50, 120000, 750, 0.15]])
y = np.array([0, 1, 0, 1])  # 0 = Loan Denied, 1 = Loan Approved

# Train a neural network classifier
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
model.fit(X, y)

# Define the objective function for generating counterfactuals
def objective_function(x_prime, x, model, target_class, lambda_param):
    distance = np.linalg.norm(x - x_prime)
    prediction = model.predict([x_prime])[0]
    classification_loss = 0 if prediction == target_class else 1
    return distance + lambda_param * classification_loss

# Generate a counterfactual explanation
def generate_counterfactual(model, x, target_class, lambda_param=0.1):
    x_prime = np.copy(x)
    result = minimize(
        objective_function,
        x_prime,
        args=(x, model, target_class, lambda_param),
        method='L-BFGS-B'
    )
    return result.x

# Example input: Applicant profile [age, income, credit score, debt-to-income ratio]
x = np.array([30, 50000, 670, 0.28])
target_class = 1  # Desired outcome: Loan approval

# Generate the counterfactual example
counterfactual = generate_counterfactual(model, x, target_class)

print("Original input:", x)
print("Counterfactual example:", counterfactual)
print("Original prediction:", model.predict([x])[0])
print("Counterfactual prediction:", model.predict([counterfactual])[0])
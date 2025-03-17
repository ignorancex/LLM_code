import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

# Define a simple LSTM model for time series prediction
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])

# Generate a synthetic time series dataset
np.random.seed(0)
X = np.random.rand(100, 10, 1)  # 100 sequences of length 10
y = (X.mean(axis=1) > 0.5).astype(int)  # Binary target based on mean value

# Train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16)

# Define the objective function for generating counterfactuals
def objective_function(x_prime, x, model, target_class, lambda_param):
    x_prime = x_prime.reshape(1, -1, 1)
    distance = np.linalg.norm(x - x_prime)
    smoothness = np.sum(np.abs(np.diff(x_prime.squeeze())))
    prediction = model.predict(x_prime)[0][0]
    classification_loss = 0 if (prediction > 0.5) == target_class else 1
    return distance + lambda_param * smoothness + 10 * classification_loss

# Generate a counterfactual for a sample sequence
def generate_counterfactual(model, x, target_class, lambda_param=0.1):
    x_prime = np.copy(x)
    result = minimize(
        objective_function,
        x_prime.flatten(),
        args=(x, model, target_class, lambda_param),
        method='L-BFGS-B'
    )
    return result.x.reshape(-1, 1)

# Example input sequence
x_sample = X[0]
target_class = 1  # Desired outcome: Change prediction to class 1

# Generate the counterfactual sequence
counterfactual = generate_counterfactual(model, x_sample, target_class)

print("Original sequence:", x_sample.flatten())
print("Counterfactual sequence:", counterfactual.flatten())
print("Original prediction:", model.predict(x_sample.reshape(1, -1, 1))[0][0])
print("Counterfactual prediction:", model.predict(counterfactual.reshape(1, -1, 1))[0][0])
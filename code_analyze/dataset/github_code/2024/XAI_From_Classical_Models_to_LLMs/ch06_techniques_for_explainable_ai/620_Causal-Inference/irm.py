import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Generate synthetic data for two environments
def generate_data(n, env_factor):
    X = np.random.randn(n, 2)
    Y = (X[:, 0] * X[:, 1] > 0).astype(int)  # Basic correlation
    Y_spurious = Y.copy()
    flip_idx = np.random.rand(n) < env_factor  # Add spurious correlation
    Y_spurious[flip_idx] = 1 - Y_spurious[flip_idx]
    return X, Y_spurious

# Environment 1 with spurious correlation factor of 0.2
X_env1, Y_env1 = generate_data(1000, env_factor=0.2)
# Environment 2 with spurious correlation factor of 0.8
X_env2, Y_env2 = generate_data(1000, env_factor=0.8)

# Reshape labels to shape (n, 1)
Y_env1 = Y_env1.reshape(-1, 1)
Y_env2 = Y_env2.reshape(-1, 1)

# Define the IRM model
class IRMModel(tf.keras.Model):
    def __init__(self):
        super(IRMModel, self).__init__()
        self.feature_extractor = Dense(10, activation='relu')
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        features = self.feature_extractor(inputs)
        output = self.classifier(features)
        return output, features

# Define IRM penalty function, using GradientTape to compute gradients
def irm_penalty(loss, features, tape):
    grad = tape.gradient(loss, features)
    penalty = tf.reduce_mean(tf.square(grad))
    return penalty

# Compile the model
model = IRMModel()
optimizer = tf.keras.optimizers.Adam()

# Training loop
for epoch in range(1000):
    with tf.GradientTape(persistent=True) as tape:
        # Process Environment 1
        pred_env1, features_env1 = model(X_env1)
        loss_env1 = tf.keras.losses.binary_crossentropy(Y_env1, pred_env1)
        loss_env1_mean = tf.reduce_mean(loss_env1)
        penalty_env1 = irm_penalty(loss_env1_mean, features_env1, tape)

        # Process Environment 2
        pred_env2, features_env2 = model(X_env2)
        loss_env2 = tf.keras.losses.binary_crossentropy(Y_env2, pred_env2)
        loss_env2_mean = tf.reduce_mean(loss_env2)
        penalty_env2 = irm_penalty(loss_env2_mean, features_env2, tape)

        # Total loss
        total_loss = loss_env1_mean + loss_env2_mean + 1.0 * (penalty_env1 + penalty_env2)

    # Compute gradients and update parameters
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape  # Free resources

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.numpy()}")

print("Training complete.")

# Evaluate the model on different environments
def evaluate_model(X, Y):
    pred, _ = model(X)
    pred_labels = (pred.numpy() > 0.5).astype(int)
    accuracy = np.mean(pred_labels == Y)
    return accuracy

acc_env1 = evaluate_model(X_env1, Y_env1)
acc_env2 = evaluate_model(X_env2, Y_env2)
print(f"Accuracy in Environment 1: {acc_env1 * 100:.2f}%")
print(f"Accuracy in Environment 2: {acc_env2 * 100:.2f}%")

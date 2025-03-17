import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "not available")

# Generate a synthetic sine wave dataset
time_steps = 100
X = np.sin(np.linspace(0, 20, time_steps))
X = X.reshape((1, time_steps, 1))  # Reshape for RNN input (batch_size, time_steps, features)

# Build a simple RNN model with 10 hidden units
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10, return_sequences=True, input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)  # Add a Dense layer for prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Run inference using the sine wave data
y_pred = model.predict(X)
hidden_states = model.predict(X, verbose=0)

# Plot the original sine wave data
plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 0], label='Original Sine Wave', color='gray', linestyle='--', linewidth=2)
plt.title("Original Sine Wave Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.show()

# Plot the predicted sine wave vs the original sine wave
plt.figure(figsize=(12, 6))
plt.plot(X[0, :, 0], label='Original Sine Wave', color='lightgray', linestyle='--', linewidth=2)
plt.plot(y_pred[0, :, 0], label='Predicted Sine Wave', color='blue', linewidth=2)
plt.title("Comparison of Original and Predicted Sine Wave")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# Plot the activations of all 10 hidden units over time
plt.figure(figsize=(14, 8))
for i in range(hidden_states.shape[-1]):
    plt.plot(hidden_states[0, :, i], label=f'Hidden Unit {i+1}', alpha=0.8)

plt.title("Activations of All Hidden Units Over Time")
plt.xlabel("Time Step")
plt.ylabel("Hidden State Activation")
plt.legend(loc='upper right', ncol=2, fontsize=10)
plt.grid()
plt.show()

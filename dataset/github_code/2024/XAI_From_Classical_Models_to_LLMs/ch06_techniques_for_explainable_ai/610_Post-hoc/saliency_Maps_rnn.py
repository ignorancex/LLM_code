import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
time_steps = 100
X = np.sin(np.linspace(0, 4 * np.pi, time_steps)) + np.random.normal(0, 0.1, time_steps)
y = np.roll(X, -1)

# Reshape data to fit LSTM input
X_input = X.reshape((1, time_steps, 1))
y_input = y.reshape((1, time_steps, 1))

# Define LSTM model to predict the entire sequence
inputs = tf.keras.Input(shape=(time_steps, 1))
lstm_output = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(lstm_output)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_input, y_input, epochs=10, verbose=0)

# Convert X and y to TensorFlow tensors
X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_input, dtype=tf.float32)

# Compute the saliency map
with tf.GradientTape() as tape:
    tape.watch(X_tensor)
    predictions = model(X_tensor)
    # Use MeanSquaredError loss object
    loss_object = tf.keras.losses.MeanSquaredError()
    loss = tf.reduce_mean(loss_object(y_tensor, predictions))

# Compute the gradients
grads = tape.gradient(loss, X_tensor)
grads = grads.numpy()[0, :, 0]  # Extract gradients for each time step

# Plot the original data, predicted data, and saliency map
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the original data and predicted data on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Data Value', color=color)
ax1.plot(X_input[0, :, 0], color='blue', label='Original Data')
ax1.plot(predictions.numpy()[0, :, 0], color='orange', label='Predicted Data')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Plot the saliency map on the right y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Gradient Magnitude', color=color)
ax2.plot(np.abs(grads), color=color, label='Saliency Map')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Original Data, Predicted Data, and Saliency Map')
plt.show()

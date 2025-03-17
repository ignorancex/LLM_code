import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(0)
time_steps = 100
X = np.sin(np.linspace(0, 2 * np.pi, time_steps)) + np.random.normal(0, 0.1, time_steps)
y = np.roll(X, -1)

# Reshape data to fit LSTM input
X_input = X.reshape((1, time_steps, 1))
y_input = y.reshape((1, time_steps, 1))

# Define the Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.score_layer = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, x):
        score = self.score_layer(x)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * x
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Build the model
inputs = tf.keras.Input(shape=(time_steps, 1))
lstm_output = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
context_vector, attention_weights = AttentionLayer()(lstm_output)
outputs = tf.keras.layers.Dense(1)(context_vector)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_input, y_input[:, -1, :], epochs=10, verbose=0)  # Adjust target values to match output shape

# Create a model to output predictions and attention weights
attention_model = tf.keras.Model(inputs=inputs, outputs=[outputs, attention_weights])

# Get predictions and attention weights
prediction, att_weights = attention_model.predict(X_input)

# Plot the original data, predicted values, and attention weights
plt.figure(figsize=(12, 6))

# Plot original data and predicted values on the left y-axis
ax1 = plt.gca()
ax1.plot(np.arange(time_steps), X, label='Original Data', color='blue')
ax1.plot(time_steps - 1, prediction[0], 'ro', label='Predicted Value')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Data Value')
ax1.legend(loc='upper left')

# Plot attention weights on the right y-axis
ax2 = ax1.twinx()
ax2.plot(np.arange(time_steps), att_weights[0, :, 0], label='Attention Weights', color='green', alpha=0.5)
ax2.set_ylabel('Attention Weights')
ax2.legend(loc='upper right')

plt.title('Original Data, Predicted Values, and Attention Weights')
plt.show()

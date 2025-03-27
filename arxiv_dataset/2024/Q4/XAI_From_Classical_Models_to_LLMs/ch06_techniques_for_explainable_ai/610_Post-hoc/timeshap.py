import numpy as np
import tensorflow as tf
import shap

# Generate synthetic time-series data
time_series_data = np.random.rand(100, 10, 1)  # 100 samples, 10 time steps, 1 feature
labels = np.random.randint(0, 2, size=(100,))

# Define a simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(time_series_data, labels, epochs=5, verbose=0)

# Define the prediction function for SHAP
def predict_fn(data):
    # Reshape input data back to 3D for the LSTM model
    return model.predict(data.reshape(-1, 10, 1)).flatten()

# Select a background dataset (subset of training data)
background_data = time_series_data[:50].reshape(50, -1)  # Flatten to 2D

# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(predict_fn, background_data)

# Select an instance to explain and flatten it
instance = time_series_data[0:1].reshape(1, -1)  # Flatten to 2D

# Compute SHAP values
shap_values = explainer.shap_values(instance, nsamples=100)

# Display SHAP values
print("SHAP values for each flattened feature:", shap_values)

# Visualization of SHAP values
import matplotlib.pyplot as plt

# Plot SHAP values for the first instance
plt.bar(range(len(shap_values[0])), shap_values[0])
plt.xlabel('Flattened Feature Index')
plt.ylabel('SHAP Value')
plt.title('SHAP Values for Time-Series Instance')
plt.show()

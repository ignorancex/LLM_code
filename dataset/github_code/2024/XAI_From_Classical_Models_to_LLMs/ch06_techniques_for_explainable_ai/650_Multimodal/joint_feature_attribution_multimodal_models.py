import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np

# Define text and image inputs
text_input = tf.keras.Input(shape=(300,), name='text_input')  # 300-d text embeddings
image_input = tf.keras.Input(shape=(512,), name='image_input')  # 512-d image features

# Define simple Dense layers for text and image features
text_features = Dense(128, activation='relu')(text_input)
image_features = Dense(128, activation='relu')(image_input)

# Concatenate text and image features
combined_features = Concatenate()([text_features, image_features])
output = Dense(1, activation='sigmoid')(combined_features)

# Build and compile the model
model = Model(inputs=[text_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Integrated Gradients function
def integrated_gradients(model, inputs, baseline, steps=50):
    alphas = np.linspace(0, 1, steps)

    # Interpolate for each input
    input_scaled_text = np.array([baseline[0] + alpha * (inputs[0] - baseline[0]) for alpha in alphas])
    input_scaled_image = np.array([baseline[1] + alpha * (inputs[1] - baseline[1]) for alpha in alphas])

    # Convert NumPy arrays to TensorFlow tensors
    input_scaled_text = tf.convert_to_tensor(input_scaled_text, dtype=tf.float32)
    input_scaled_image = tf.convert_to_tensor(input_scaled_image, dtype=tf.float32)

    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        tape.watch([input_scaled_text, input_scaled_image])
        predictions = model([input_scaled_text, input_scaled_image])
    gradients = tape.gradient(predictions, [input_scaled_text, input_scaled_image])

    # Calculate average gradients and Integrated Gradients
    avg_gradients_text = tf.reduce_mean(gradients[0], axis=0).numpy()
    avg_gradients_image = tf.reduce_mean(gradients[1], axis=0).numpy()

    integrated_gradients_text = (inputs[0] - baseline[0]) * avg_gradients_text
    integrated_gradients_image = (inputs[1] - baseline[1]) * avg_gradients_image

    return integrated_gradients_text, integrated_gradients_image

# Example usage
text_sample = np.random.rand(300)
image_sample = np.random.rand(512)
baseline_text = np.zeros(300)
baseline_image = np.zeros(512)

inputs = [text_sample, image_sample]
baseline = [baseline_text, baseline_image]

# Compute Integrated Gradients
attributions_text, attributions_image = integrated_gradients(model, inputs, baseline)
print("Integrated Gradients for text features:", attributions_text)
print("Integrated Gradients for image features:", attributions_image)

# Visualize Integrated Gradients
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot for text features
plt.subplot(1, 2, 1)
plt.plot(attributions_text)
plt.title("Integrated Gradients for Text Features")
plt.xlabel("Feature Index")
plt.ylabel("Attribution")

# Plot for image features
plt.subplot(1, 2, 2)
plt.plot(attributions_image)
plt.title("Integrated Gradients for Image Features")
plt.xlabel("Feature Index")
plt.ylabel("Attribution")

plt.tight_layout()
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define a simple CNN model in TensorFlow/Keras
class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Load the MNIST dataset
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images[..., tf.newaxis] / 255.0  # Rescale to [0, 1]

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Train the model (one epoch for simplicity)
model.fit(train_images, train_labels, batch_size=64, epochs=1)

# Select a sample image and baseline for Integrated Gradients
sample_image = train_images[0:1]  # Shape (1, 28, 28, 1)
baseline = tf.zeros_like(sample_image)

# Function to calculate Integrated Gradients
def compute_integrated_gradients(model, input_image, baseline, target_class_idx, m_steps=50):
    # Generate interpolated images between baseline and input
    interpolated_images = [
        baseline + (float(i) / m_steps) * (input_image - baseline)
        for i in range(m_steps + 1)
    ]
    interpolated_images = tf.concat(interpolated_images, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        # Get model predictions for interpolated images
        predictions = model(interpolated_images)
        target_predictions = predictions[:, target_class_idx]

    # Compute gradients between predictions and interpolated images
    grads = tape.gradient(target_predictions, interpolated_images)

    # Average gradients and compute attributions
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (input_image - baseline) * avg_grads
    return integrated_grads

# Compute attributions using Integrated Gradients
target_class = train_labels[0]  # The true class for the sample image
attributions = compute_integrated_gradients(model, sample_image, baseline, target_class)

# Visualize the attributions
attributions = attributions.numpy().squeeze()
plt.imshow(attributions, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Integrated Gradients Attribution for MNIST Prediction")
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Function to compute saliency map
def compute_saliency_map(model, input_image, label):
    input_image = tf.convert_to_tensor(input_image[np.newaxis, ...])
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image)
        loss = predictions[0, label]
    gradient = tape.gradient(loss, input_image)
    saliency_map = tf.reduce_max(tf.abs(gradient), axis=-1).numpy()[0]
    return saliency_map

# Test robustness of the saliency map
original_image = x_test[0]
perturbed_image = original_image + 0.1 * np.random.normal(size=original_image.shape)

# Compute saliency maps
saliency_original = compute_saliency_map(model, original_image, y_test[0])
saliency_perturbed = compute_saliency_map(model, perturbed_image, y_test[0])

# Compute robustness score
robustness_score = 1 - np.linalg.norm(saliency_original - saliency_perturbed) / np.linalg.norm(original_image - perturbed_image)
print(f"Robustness Score: {robustness_score:.4f}")

# Display the saliency maps
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Saliency Map")
plt.imshow(saliency_original, cmap='hot')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Perturbed Saliency Map")
plt.imshow(saliency_perturbed, cmap='hot')
plt.axis('off')
plt.show()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, batch_size=64)

# Function to compute the saliency map
def compute_saliency_map(model, image, target_class):
    image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[0, target_class]

    # Compute the gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, image)
    saliency = tf.abs(gradient)[0]

    # Normalize the saliency map
    saliency = saliency.numpy().squeeze()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency

# Select a sample image and compute the saliency map
sample_image = test_images[0]
target_class = np.argmax(model.predict(sample_image[np.newaxis, ...]))
saliency_map = compute_saliency_map(model, sample_image, target_class)

# Visualize the original image and its saliency map
plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title("Saliency Map")
plt.colorbar()
plt.show()
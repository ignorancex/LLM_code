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

# Function to compute SmoothGrad
def smoothgrad(image, model, target_class, num_samples=50, noise_level=0.1):
    grads = []
    for _ in range(num_samples):
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        noisy_image = tf.convert_to_tensor(noisy_image[np.newaxis, ...], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(noisy_image)
            prediction = model(noisy_image)
            loss = prediction[0, target_class]

        gradient = tape.gradient(loss, noisy_image)
        grads.append(gradient.numpy().squeeze())

    # Average the gradients
    smooth_grad = np.mean(grads, axis=0)
    return smooth_grad

# Select a sample image and compute SmoothGrad attributions
sample_image = test_images[0]
target_class = np.argmax(model.predict(sample_image[np.newaxis, ...]))
attributions = smoothgrad(sample_image, model, target_class)

# Visualize the attributions
plt.imshow(attributions, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("SmoothGrad Attribution for MNIST Prediction")
plt.show()
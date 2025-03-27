import tensorflow as tf
import numpy as np

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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Function to compute gradient-based explanations
def compute_gradients(model, input_image, label):
    input_image = tf.convert_to_tensor(input_image[np.newaxis, ...])
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image)
        loss = predictions[0, label]
    gradient = tape.gradient(loss, input_image).numpy()[0]
    return gradient

# Consistency Analysis
image1 = x_test[0]
image2 = x_test[1]  # A similar image
grad1 = compute_gradients(model, image1, y_test[0])
grad2 = compute_gradients(model, image2, y_test[1])
ecs = np.dot(grad1.flatten(), grad2.flatten()) / (np.linalg.norm(grad1) * np.linalg.norm(grad2))
print(f"Explanation Consistency Score (ECS): {ecs:.4f}")

# Stability Analysis
perturbed_image = image1 + 0.1 * np.random.normal(size=image1.shape)
grad_perturbed = compute_gradients(model, perturbed_image, y_test[0])
ess = 1 - np.linalg.norm(grad1 - grad_perturbed) / np.linalg.norm(image1 - perturbed_image)
print(f"Explanation Stability Score (ESS): {ess:.4f}")
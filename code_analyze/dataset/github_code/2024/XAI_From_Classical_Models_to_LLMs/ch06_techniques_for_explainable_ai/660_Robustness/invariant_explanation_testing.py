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

# Split test data into two environments: even and odd digits
even_digits = x_test[y_test % 2 == 0]
odd_digits = x_test[y_test % 2 == 1]

# Compute gradient explanations for both environments
grad_even = compute_gradients(model, even_digits[0], y_test[0])
grad_odd = compute_gradients(model, odd_digits[0], y_test[1])

# Calculate Explanation Invariance Score (EIS)
eis = 1 - np.linalg.norm(grad_even - grad_odd) / (np.linalg.norm(grad_even) + np.linalg.norm(grad_odd))
print(f"Explanation Invariance Score (EIS): {eis:.4f}")
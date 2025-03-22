import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

# Define a simple neural network model using Functional API to avoid warnings
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, batch_size=64)

# Function to generate an optimization-based counterfactual
def generate_counterfactual(model, image, target_class, num_steps=100, learning_rate=0.01, lambda_param=0.1):
    # Create a variable for the counterfactual image
    counterfactual = tf.Variable(image, dtype=tf.float32)

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            # Compute the loss: distance loss + prediction loss
            distance_loss = tf.reduce_mean(tf.abs(counterfactual - image))
            prediction = model(counterfactual)
            # Convert target_class to tensor
            target_class_tensor = tf.convert_to_tensor([target_class], dtype=tf.int32)
            classification_loss = tf.keras.losses.sparse_categorical_crossentropy(target_class_tensor, prediction)
            total_loss = distance_loss + lambda_param * classification_loss

        # Compute gradients and update the counterfactual image
        gradients = tape.gradient(total_loss, counterfactual)
        optimizer.apply_gradients([(gradients, counterfactual)])

        # Clip the pixel values to maintain valid image range
        counterfactual.assign(tf.clip_by_value(counterfactual, 0.0, 1.0))

    return counterfactual.numpy()

# Select a sample image and generate a counterfactual
sample_image = test_images[0:1]
original_prediction = model.predict(sample_image)
original_label = np.argmax(original_prediction, axis=1)[0]
target_label = (original_label + 1) % 10

counterfactual_image = generate_counterfactual(model, sample_image, target_label)

# Display the original and counterfactual images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"Original: {original_label}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(counterfactual_image.squeeze(), cmap='gray')
plt.title(f"Counterfactual: {target_label}")
plt.axis('off')

plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from scipy.optimize import minimize

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# Define a simple classifier model
def create_classifier():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)  # Logits output
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

classifier = create_classifier()
classifier.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the classifier
classifier.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels),
    batch_size=128
)

# Define a simple generator model
def create_generator(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 128, activation='relu')(inputs)
    x = tf.keras.layers.Reshape((7, 7, 128))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

latent_dim = 100
generator = create_generator(latent_dim)

# Note: In practice, you should train the generator as part of a GAN.
# For this example, we'll assume the generator is already trained.

# Select a sample image from the test set
sample_image = test_images[0:1]
original_prediction = classifier.predict(sample_image)
original_label = np.argmax(original_prediction, axis=1)[0]

# Define the target class (different from the original prediction)
target_label = (original_label + 1) % 10

# Define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define a function to generate counterfactuals
def generate_counterfactual(z):
    z = tf.convert_to_tensor(z.reshape(1, -1), dtype=tf.float32)
    with tf.GradientTape():
        generated_image = generator(z)
        prediction = classifier(generated_image)
        # Convert target_label to tensor
        target_label_tensor = tf.convert_to_tensor([target_label], dtype=tf.int32)
        similarity_loss = tf.norm(generated_image - sample_image)
        classification_loss = loss_fn(target_label_tensor, prediction)
        total_loss = similarity_loss + 0.1 * classification_loss
    return total_loss.numpy().astype(np.float64)

# Initialize the latent vector and optimize it
z_initial = np.random.normal(size=(latent_dim,))
result = minimize(
    generate_counterfactual,
    z_initial,
    method='L-BFGS-B',
    options={'maxiter': 100}
)

# Generate the counterfactual image
z_optimized = result.x
counterfactual_image = generator.predict(z_optimized.reshape(1, -1))

# Display the original and counterfactual images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"Original: {original_label}")
plt.axis('off')

plt.subplot(1, 2, 2)
# Removed .numpy() since counterfactual_image is already a NumPy array
plt.imshow(counterfactual_image.squeeze(), cmap='gray')
plt.title(f"Counterfactual: {target_label}")
plt.axis('off')

plt.show()

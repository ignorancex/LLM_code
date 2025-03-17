import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

# Define and train the classifier model (if not already trained)
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Save the trained model
model.save('mnist_classifier.h5')

# Load the model (optional if already in memory)
model = tf.keras.models.load_model('mnist_classifier.h5')

# Function to generate diverse counterfactuals
def generate_diverse_counterfactuals(model, image, target_class, num_counterfactuals=3, num_steps=100, learning_rate=0.01, lambda_1=0.1, lambda_2=0.05):
    counterfactuals = []

    for _ in range(num_counterfactuals):
        # Initialize the counterfactual image as a copy of the original
        counterfactual = tf.Variable(image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate)

        for step in range(num_steps):
            with tf.GradientTape() as tape:
                # Compute the prediction loss
                prediction = model(counterfactual)
                target_class_tensor = tf.convert_to_tensor([target_class], dtype=tf.int32)
                classification_loss = tf.keras.losses.sparse_categorical_crossentropy(target_class_tensor, prediction)

                # Compute the similarity loss
                distance_loss = tf.reduce_mean(tf.abs(counterfactual - image))

                # Compute the diversity loss (based on difference from previous counterfactuals)
                diversity_loss = 0
                if counterfactuals:
                    for prev_cf in counterfactuals:
                        diversity_loss += tf.reduce_mean(tf.abs(counterfactual - prev_cf))
                    diversity_loss /= len(counterfactuals)  # Normalize by the number of counterfactuals

                # Total loss function
                total_loss = distance_loss + lambda_1 * classification_loss + lambda_2 * diversity_loss

            # Update the counterfactual image
            gradients = tape.gradient(total_loss, counterfactual)
            optimizer.apply_gradients([(gradients, counterfactual)])
            counterfactual.assign(tf.clip_by_value(counterfactual, 0.0, 1.0))

        # Add the optimized counterfactual to the list
        counterfactuals.append(counterfactual.numpy())

    return counterfactuals

# Select a sample image and generate counterfactuals
sample_image = test_images[0:1]
original_prediction = model.predict(sample_image)
original_label = np.argmax(original_prediction, axis=1)[0]
target_label = (original_label + 1) % 10  # Set the desired target class

print(f"Original label: {original_label}, Target label: {target_label}")

# Generate diverse counterfactuals
counterfactuals = generate_diverse_counterfactuals(model, sample_image, target_label)

# Display the generated counterfactuals
plt.figure(figsize=(12, 4))
for i, cf_image in enumerate(counterfactuals):
    plt.subplot(1, len(counterfactuals), i + 1)
    plt.imshow(cf_image.squeeze(), cmap='gray')
    plt.title(f"Counterfactual {i+1}")
    plt.axis('off')
plt.show()

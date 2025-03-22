import tensorflow as tf
import numpy as np

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target
sensitive_attribute = np.random.choice([0, 1], size=(1000,))  # Gender (0 = male, 1 = female)

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Function to compute fairness-aware attributions
def compute_attributions(model, X, sensitive_attr):
    gradients_male = []
    gradients_female = []

    for i, x in enumerate(X):
        # Convert input to tf.Tensor
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            prediction = model(tf.expand_dims(x_tensor, axis=0))
        gradient = tape.gradient(prediction, x_tensor).numpy()

        if sensitive_attr[i] == 0:  # Male
            gradients_male.append(gradient)
        else:  # Female
            gradients_female.append(gradient)

    avg_grad_male = np.mean(gradients_male, axis=0)
    avg_grad_female = np.mean(gradients_female, axis=0)
    attribution_disparity = np.abs(avg_grad_male - avg_grad_female)

    return avg_grad_male, avg_grad_female, attribution_disparity

# Compute attributions and disparity
avg_grad_male, avg_grad_female, attribution_disparity = compute_attributions(model, X, sensitive_attribute)

print("Average gradient attributions (male):", avg_grad_male)
print("Average gradient attributions (female):", avg_grad_female)
print("Attribution disparity:", attribution_disparity)

import matplotlib.pyplot as plt

# Visualization of attribution disparity
plt.bar(range(10), attribution_disparity)
plt.xlabel('Feature Index')
plt.ylabel('Attribution Disparity')
plt.title('Feature Attribution Disparity between Male and Female')
plt.show()

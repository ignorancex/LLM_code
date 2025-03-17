import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "not available")

# Load a pre-trained VGG16 model (without the fully connected layers)
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Load and preprocess the input image
image_path = 'Ch04/cat.jpg'  # Ensure 'cat.jpg' is in the directory
try:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    exit()

image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = tf.keras.applications.vgg16.preprocess_input(image_array)

# Define a model that outputs the feature maps of the first convolutional layer
layer_name = 'block1_conv1'
feature_map_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Generate the feature maps for the input image
feature_maps = feature_map_model.predict(image_array)

# Check the shape of the feature maps
print("Feature map shape:", feature_maps.shape)

# Visualize the first 16 feature maps
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < feature_maps.shape[-1]:
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.tight_layout()
plt.show()

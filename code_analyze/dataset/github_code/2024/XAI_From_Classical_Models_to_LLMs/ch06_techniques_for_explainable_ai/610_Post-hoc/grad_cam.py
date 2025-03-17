import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2  # 新增 OpenCV 用於調整 heatmap 大小

# Load a pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess the input image
image_path = 'Ch04/Images/cat.jpg'
image = load_img(image_path, target_size=(224, 224))

# Convert the image to an array and preprocess it
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

# Get the model prediction
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions[0])

# Function to compute Grad-CAM heatmap
def compute_gradcam(model, image_array, class_idx, layer_name='block5_conv3'):
    # Create a model that maps the input image to the activations of the last convolutional layer
    # and the model's output
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_array)
        loss = predictions[:, class_idx]

    # Compute gradients with respect to the convolutional output
    grads = tape.gradient(loss, conv_output)

    # Compute the mean intensity of the gradients for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Extract the feature maps from the convolutional layer output
    conv_output = conv_output[0]

    # Compute the weighted sum of the feature maps
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    # Apply ReLU and normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Generate the Grad-CAM heatmap
heatmap = compute_gradcam(model, image_array, predicted_class)

# Resize heatmap to match the input image size
heatmap = cv2.resize(heatmap, (224, 224))  # 使用 OpenCV 調整大小
heatmap = np.uint8(255 * heatmap)  # 將 heatmap 轉換為 0-255 的範圍

# 將 heatmap 轉換為彩色映射
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 疊加 heatmap 和原始圖像
superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)

# Display the original image, heatmap, and overlay
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Display original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title("Original Image")

# Display the heatmap only
ax[1].imshow(heatmap, cmap='jet')
ax[1].axis('off')
ax[1].set_title("Grad-CAM Heatmap")

# Display the overlay image
ax[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
ax[2].axis('off')
ax[2].set_title("Overlay Image")

# Show the plot
plt.tight_layout()
plt.show()

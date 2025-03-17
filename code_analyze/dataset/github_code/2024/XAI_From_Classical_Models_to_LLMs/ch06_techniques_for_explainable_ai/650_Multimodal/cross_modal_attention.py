import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, Concatenate, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Set the model dimension
d_model = 64

# Define text and image input layers
text_input = tf.keras.Input(shape=(100, 300), name='text_input')  # 100 tokens, 300-d embeddings
image_input = tf.keras.Input(shape=(49, 512), name='image_input')  # 7x7 image patches, 512-d features

# Project inputs to a common dimension
text_features = Dense(d_model)(text_input)    # Shape: (batch_size, 100, 64)
image_features = Dense(d_model)(image_input)  # Shape: (batch_size, 49, 64)

# Self-attention on text features
text_self_attention = MultiHeadAttention(num_heads=8, key_dim=d_model)
text_attention_output = text_self_attention(
    query=text_features, value=text_features, key=text_features
)  # Shape: (batch_size, 100, 64)

# Self-attention on image features
image_self_attention = MultiHeadAttention(num_heads=8, key_dim=d_model)
image_attention_output = image_self_attention(
    query=image_features, value=image_features, key=image_features
)  # Shape: (batch_size, 49, 64)

# Cross-modal attention from text to image features
cross_modal_attention = MultiHeadAttention(num_heads=8, key_dim=d_model)
cross_attention_output, cross_attention_scores = cross_modal_attention(
    query=text_attention_output,
    value=image_attention_output,
    key=image_attention_output,
    return_attention_scores=True
)  # cross_attention_output shape: (batch_size, 100, 64)
# cross_attention_scores shape: (batch_size, 8, 100, 49)

# Average the attention scores over the heads using a Lambda layer
average_attention_scores = Lambda(lambda x: tf.reduce_mean(x, axis=1))(cross_attention_scores)
# Shape: (batch_size, 100, 49)

# Pooling over the sequence length to get fixed-size representations
text_representation = GlobalAveragePooling1D()(text_attention_output)          # Shape: (batch_size, 64)
image_representation = GlobalAveragePooling1D()(image_attention_output)        # Shape: (batch_size, 64)
cross_attention_representation = GlobalAveragePooling1D()(cross_attention_output)  # Shape: (batch_size, 64)

# Combine the representations
combined_representation = Concatenate()([
    text_representation,
    image_representation,
    cross_attention_representation
])  # Shape: (batch_size, 192)

# Output layer
output = Dense(1, activation='sigmoid')(combined_representation)

# Build and compile the model
model = Model(inputs=[text_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create sample input data
text_sample = np.random.rand(1, 100, 300).astype(np.float32)
image_sample = np.random.rand(1, 49, 512).astype(np.float32)

# Perform model prediction
prediction = model.predict([text_sample, image_sample])
print("Prediction:", prediction)

# Build a model to output the attention scores for visualization
attention_model = Model(inputs=[text_input, image_input], outputs=average_attention_scores)

# Get the attention scores
attention_scores = attention_model.predict([text_sample, image_sample])  # Shape: (1, 100, 49)

# Visualize the cross-modal attention weights
plt.figure(figsize=(12, 8))
plt.imshow(attention_scores[0], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Cross-modal Attention Weights (Text to Image Features)")
plt.xlabel("Image Feature Index (49 patches)")
plt.ylabel("Text Token Index (100 tokens)")
plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Attention, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Define text and image inputs
text_input = tf.keras.Input(shape=(100, 300), name='text_input')  # 100 tokens, 300-d embeddings
image_input = tf.keras.Input(shape=(49, 512), name='image_input')  # 7x7 image patches, 512-d features

# Text attention mechanism
text_query = Dense(64, activation='tanh')(text_input)
text_key = Dense(64, activation='tanh')(text_input)
text_value = Dense(64, activation='tanh')(text_input)
text_attention = Attention()([text_query, text_value, text_key])

# Reduce mean for text attention
text_representation = Lambda(lambda x: tf.reduce_mean(x, axis=1))(text_attention)

# Image attention mechanism
image_query = Dense(64, activation='tanh')(image_input)
image_key = Dense(64, activation='tanh')(image_input)
image_value = Dense(64, activation='tanh')(image_input)
image_attention = Attention()([image_query, image_value, image_key])

# Reduce mean for image attention
image_representation = Lambda(lambda x: tf.reduce_mean(x, axis=1))(image_attention)

# Concatenate text and image representations
combined_representation = Concatenate()([text_representation, image_representation])
output = Dense(1, activation='sigmoid')(combined_representation)

# Build and compile the model
model = Model(inputs=[text_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Save the model plot as a PNG image
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

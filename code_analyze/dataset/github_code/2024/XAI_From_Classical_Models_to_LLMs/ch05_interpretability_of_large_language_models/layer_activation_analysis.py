import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt

# Load pretrained BERT model and tokenizer with hidden states output enabled
model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to extract and analyze layer-wise activations
def extract_and_analyze_activations(model, inputs):
    outputs = model(inputs)
    hidden_states = outputs.hidden_states  # Extract all hidden states
    layer_means = [tf.reduce_mean(state).numpy() for state in hidden_states]  # Compute mean activation
    return hidden_states, layer_means

# Example usage with a sample input sentence
input_data = tokenizer("The cat sat on the mat.", return_tensors='tf', padding=True, truncation=True)
input_ids = input_data['input_ids']

# Extract activations and compute mean activations for each layer
layer_outputs, layer_means = extract_and_analyze_activations(model, input_ids)

# Print the number of layers and the shape of activations from the first layer
print("Number of layers analyzed:", len(layer_outputs))
print("Shape of activations from the first layer:", layer_outputs[0].shape)

# Plot mean activations across layers
plt.figure(figsize=(10, 6))
plt.plot(range(len(layer_means)), layer_means, marker='o', color='blue')
plt.title("Mean Activations Across BERT Layers")
plt.xlabel("Layer")
plt.ylabel("Mean Activation Value")
plt.grid(True)
plt.show()

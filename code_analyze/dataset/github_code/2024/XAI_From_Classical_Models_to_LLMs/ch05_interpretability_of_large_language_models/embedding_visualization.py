import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load pretrained BERT model and tokenizer
model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a list of words to encode
words = ["king", "queen", "man", "woman"]

# Tokenize words and obtain embeddings
inputs = tokenizer(words, return_tensors='tf', padding=True, truncation=True)
outputs = model(inputs['input_ids'])[0].numpy()

# Compute the mean embeddings for each word
mean_embeddings = outputs.mean(axis=1)

# Perform PCA to reduce embeddings to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(mean_embeddings)

# Plot the 2D embeddings
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue')

# Annotate the plot with word labels
for i, word in enumerate(words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)

plt.title("PCA Visualization of BERT Word Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

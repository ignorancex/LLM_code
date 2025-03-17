import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load pretrained BERT model and tokenizer
model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a list of words and their corresponding part-of-speech labels
words = ["cat", "run", "dog", "jump"]  # Example words
labels = [0, 1, 0, 1]  # Labels: 0 for noun, 1 for verb

# Tokenize the words and obtain embeddings
inputs = tokenizer(words, return_tensors='tf', padding=True, truncation=True)
outputs = model(inputs['input_ids'])[0].numpy()

# Use mean embeddings as features for the classifier
features = outputs.mean(axis=1)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(features, labels)

# Make predictions
predictions = classifier.predict(features)

# Evaluate the classifier
accuracy = accuracy_score(labels, predictions)
print(f"Probing Task Accuracy: {accuracy:.2f}")

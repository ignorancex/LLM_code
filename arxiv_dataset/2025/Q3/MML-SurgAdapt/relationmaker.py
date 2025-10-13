import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os

# Step 1: Read the text files and convert them to lists
def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()  # Read lines and strip newline characters
    return lines

file1_path = 'cholec/cholec_labels.txt'  # Replace with your actual file path
file2_path = 'cholec/cholec_super_labels.txt'  # Replace with your actual file path

list1 = read_file_to_list(file1_path)
list2 = read_file_to_list(file2_path)

# Step 2: Concatenate the two lists
combined_list = list1 + list2

# Step 3: Load the pre-trained Word2Vec model (Google News)
# You can download the pre-trained model from: https://code.google.com/archive/p/word2vec/
model_path = 'GoogleNews-vectors-negative300.bin.gz'  # Replace with your actual path
if not os.path.exists(model_path):
    raise FileNotFoundError("Download the model from GoogleNews-vectors-negative300.bin.gz and place it in the correct path.")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Function to compute the average word vector for a sentence
def get_sentence_vector(sentence, model):
    words = sentence.split()  # Split sentence into words
    word_vectors = [model[word] for word in words if word in model]  # Get vectors for words in model
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Return a zero vector if no word is found
    return np.mean(word_vectors, axis=0)

# Step 4: Calculate sentence embeddings for each sentence in the list
sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in combined_list])

# Step 5: Compute cosine similarity matrix for the sentence embeddings
similarity_matrix = cosine_similarity(sentence_vectors)

# Step 6: Save the similarity matrix as a .npy file
output_file_path = 'cholec/word2vec_similarity_matrix.npy'
np.save(output_file_path, similarity_matrix)

print(f"Word2Vec similarity matrix saved to {output_file_path}")
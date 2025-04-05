import numpy as np
from scipy.spatial import distance

def vcosine(u, v):
    return abs(1 - distance.cdist(u, v, 'cosine'))

def cosine(u, v):
    return abs(1 - distance.cosine(u, v))

def rescale(a):
    maximum = np.max(a)
    minimum = np.min(a)
    return (a - minimum) / (maximum - minimum)

def normalize(matrix):
    for row in matrix:
        row_sum = np.sum(row)
        if row_sum != 0:
            row /= row_sum
    return matrix

def biased_textrank(texts_embeddings, bias_embedding, damping_factor=0.8, similarity_threshold=0.8, biased=True):
    matrix = vcosine(texts_embeddings, texts_embeddings)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < similarity_threshold] = 0

    matrix = normalize(matrix)

    if biased:
        bias_weights = vcosine(bias_embedding, texts_embeddings)
        bias_weights = rescale(bias_weights)
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights
    else:
        scaled_matrix = damping_factor * matrix + (1 - damping_factor)

    scaled_matrix = normalize(scaled_matrix)
    # scaled_matrix = rescale(scaled_matrix)

    #print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    iterations = 80
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks

def select_top_k_texts_preserving_order(texts, ranking, k, id=0):
    texts_sorted = sorted(zip(texts, ranking), key=lambda item: item[1]) #, reverse=True) #sort in ascending
    if id == 1: print(texts_sorted)
    top_texts = texts_sorted[:k]
    top_texts = [t[0] for t in top_texts]
    
    return top_texts


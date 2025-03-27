import faiss
import numpy as np

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def knn(training_features, eval_features, k=500):
    model = faiss.IndexFlatL2(training_features.shape[1])
    model.add(normalizer(training_features))
    dist, _ = model.search(normalizer(eval_features), k)
    return -dist[:, -1]
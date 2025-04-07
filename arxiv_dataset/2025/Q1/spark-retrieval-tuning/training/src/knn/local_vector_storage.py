import numpy as np
import torch

class LocalVectorDBClient():
    def __init__(self):
        self.data = {}

    def get(self, id):
        return self.data.get(id)

    def create(self, id, value, embedding=None):
        if embedding is None:
            embedding = value["embedding"]
        
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
            value["embedding"] = embedding
        self.data[id] = value

    def update(self, id, value):
        self.data[id] = value

    def delete(self, id):
        if id in self.data:
            del self.data[id]

    def search(self, query_embedding, top_k):
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding)


        # Compute cosine similarity between query and all documents
        scores = {}
        for id, value in self.data.items():
            # print("VECTOR: ", value["embedding"], "QUERY: ", query_embedding)
            if query_embedding.shape != value["embedding"].shape:
                raise ValueError("Query embedding does not match the dimension of the stored embeddings", query_embedding.shape, value["embedding"].shape)

            scores[id] = self._compute_cosine_similarity(query_embedding, value["embedding"])
            # print("ID and score:", id, "==>", scores[id])
        # Sort by similarity and return top k
        top_chunk_ids = self._get_top_k(scores, top_k)
        return [(self.data[id], score) for id, score in top_chunk_ids]

    def _get_top_k(self, scores, top_k):
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _compute_cosine_similarity(self, embedding1, embedding2):
        # compute cosine similarity between two embeddings
        cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).to("cpu")
        return cosine_sim.item()

    def __len__(self):
        return len(self.data)


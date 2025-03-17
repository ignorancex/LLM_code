from typing import List, Optional
import faiss
import torch
from labml import lab, monit
from embeddings import ChunkEmbeddings
import numpy as np
class RetroIndex:
    def __init__(self, device, index_filepath, config_json,embedding_type="multi-qa-mpnet-base-dot-v1"):
        self.n_neighbors, self.chunk_len, self.exclude_neighbor_span, self.nprobe, self.n_extra = config_json["n_neighbors"],  config_json["chunk_len"], config_json["exclude_neighbor_span"], config_json["n_probe"], config_json["n_extra"]
        normalize_bert_bool = True if (config_json["normalize_bert"] == "True") else False
        gpt2_bool = True if (config_json["gpt2"] == "True") else False
        self.emb = ChunkEmbeddings(torch.device(device), normalize_bert_bool, gpt2_bool,embedding_type=embedding_type)
        with monit.section('Load index'):
            self.index = faiss.read_index(index_filepath)
            self.index.nprobe = self.nprobe
    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        indices = []
        for i,n in enumerate(neighbor_offsets):
            if n < offset - (self.chunk_len + self.exclude_neighbor_span) or n > offset + (self.chunk_len + self.exclude_neighbor_span):
                indices.append(i)
        return indices
    def __call__(self, query_chunks: List[str], offsets: Optional[List[int]]):
        emb = self.emb(query_chunks).cpu()
        distance, neighbor_chunk_offsets = self.index.search(emb.numpy(), self.n_neighbors + self.n_extra)
        if offsets is not None:
            neighbor_chunk_offsets_filtered_indices = [self.filter_neighbors(off, n_off) for off, n_off in zip(offsets, neighbor_chunk_offsets)]
            neighbor_chunk_offsets_res = []
            distance_res = []
            assert(len(neighbor_chunk_offsets) == len(neighbor_chunk_offsets_filtered_indices))
            for i in range(len(neighbor_chunk_offsets_filtered_indices)): #for each chunk
                neighbor_chunk_offsets_res.append([])
                distance_res.append([])
                for j in neighbor_chunk_offsets_filtered_indices[i]: #for all the neighbors that made it.
                    if len(neighbor_chunk_offsets_res[-1]) < self.n_neighbors:
                        neighbor_chunk_offsets_res[-1].append(neighbor_chunk_offsets[i][j])
                        distance_res[-1].append(distance[i][j])
                    else:
                        break
            distance_res = np.array(distance_res)
            neighbor_chunk_offsets_res = np.array(neighbor_chunk_offsets_res)
            assert(type(distance) == type(distance_res))
            assert(len(distance) == len(distance_res))
            assert(type(neighbor_chunk_offsets) == type(neighbor_chunk_offsets_res))
            assert(len(neighbor_chunk_offsets) == len(neighbor_chunk_offsets_res))
            return distance_res, neighbor_chunk_offsets_res
        else:
            return distance, neighbor_chunk_offsets
if __name__ == '__main__':
    pass    
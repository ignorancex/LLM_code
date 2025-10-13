import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from faiss_indexers import DenseFlatIndexer
import numpy as np

resume_index = DenseFlatIndexer()

class Retriever(object):
    def __init__(self, vector_path, profile_path):
        index = DenseFlatIndexer()
        index.init_index(768)
        self.index = index
        
        self.index_encoded_data(vector_path)
        self.init_docs(profile_path)

    def init_docs(self, path):
        with open(path, 'r') as f:
            contexts = json.load(f)
        profile_dict = dict()
        for item in contexts:
            profile_dict[item['id']] = item
        self.profile = profile_dict

    def index_encoded_data(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.index.index_data(data)
        print("Data indexing completed.")
    
    def get_top_docs(self, query_vectors: np.array, top_docs: int = 10) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        targets, scores = self.index.search_knn(query_vectors, top_docs)[0]
        #logger.info("index search time: %f sec.", time.time() - time0)
        #self.index = None
        results = []
        for i in range(top_docs):
            id = targets[i]
            item = self.profile[id]
            item['score'] = scores[i]
            results.append(item)
        return results
        

def main():
    
    jd_path = ""#TODO_output_path
    jd_vector = ""#TODO_input_path output_vector_path
    
    jd_retriever = Retriever(jd_vector, jd_path)
    model = SentenceTransformer('moka-ai/m3e-base')
    while True:
        query = input("#query_of_retriever")
        query_vector = model.encode([query])
        results = jd_retriever.get_top_docs(query_vector,3)
        for i in range(3):
            item = results[i]
            print(item['profile'])

if __name__ == '__main__':
    main()
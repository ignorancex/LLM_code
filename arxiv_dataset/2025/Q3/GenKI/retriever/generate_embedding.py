from sentence_transformers import SentenceTransformer
import json
import pickle
from tqdm import tqdm
def generate_jd_embedding(path):
    model = SentenceTransformer('moka-ai/m3e-base')
    with open(path,'r') as f:
        data = json.load(f)
    ids = []
    contexts = []
    results = []
    n = len(data)
    bsz = 32
    i = 0
    for  item in tqdm(data):
        i +=1
        job_id = item['id']
        info = item['profile']
        ids.append(job_id)
        contexts.append(info)
        if i % bsz == 0:
            embeddings = model.encode(contexts)
            for id, emb in zip(ids,embeddings):
                results.append((id, emb))
            contexts = []
            ids = []
    embeddings = model.encode(contexts)
    for id, emb in zip(ids,embeddings):
        results.append((id, emb))
    with open("", "wb") as f: #output_vector_path
        pickle.dump(results, f)


if __name__ == '__main__':
    jd_path = "" #input_path
    generate_jd_embedding(jd_path)
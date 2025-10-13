import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from openai import OpenAI
from tqdm import tqdm

df = pd.read_json('ivanchyk.json')
api_key_v1=""
client = OpenAI(api_key=api_key_v1)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   response = client.embeddings.create(input=[text], model=model)
   return response.data[0].embedding

batch_size = 100
embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
   batch_texts = df['ukrainian'][i:i+batch_size].tolist()
   batch_embeddings = [get_embedding(text, model='text-embedding-3-large') for text in batch_texts]
   embeddings.extend(batch_embeddings)

df['ada_embedding'] = embeddings
embeddings = np.array(df['ada_embedding'].tolist())
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, 'output/faiss_index_emb_ivanchyk.bin')


import os
import numpy as np

from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim

PATH = os.path.dirname(__file__)

model = SBert('roberta-large-nli-stsb-mean-tokens')

def sbert_similarity(list1, list2):
    embeddings1 = model.encode(list1)
    embeddings2 = model.encode(list2)
    cosine_similarity = cos_sim(embeddings1, embeddings2)
    return cosine_similarity


def get_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            print(file)
            with open(PATH+'models.txt', 'a+') as f1:
                f1.write(file)
            files.append(os.path.join(path, file))
    return files


def get_docs(files):
    docs = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            docs.append(f.read())
    return docs


def get_sbert_matrix(docs):
    sbert_matrix = []
    for i in range(len(docs)):
        sbert_matrix.append([])
        for j in range(len(docs)):
            if i == j:
                sbert_matrix[i].append(1)
            elif i > j:
                sbert_matrix[i].append(sbert_matrix[j][i])
            else:
                cosine_sim = sbert_similarity(docs[i], docs[j])
                sbert_matrix[i].append(cosine_sim)

    return sbert_matrix


path = os.path.dirname(__file__)
files = get_files(path)
docs = get_docs(files)
tfidf_matrix = get_sbert_matrix(docs)
print(tfidf_matrix)
output = np.array(tfidf_matrix)
np.savetxt("matrix.txt", output, delimiter='\t')

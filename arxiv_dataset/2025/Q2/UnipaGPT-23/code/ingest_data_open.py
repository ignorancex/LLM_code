from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import argparse
import json
import os

def load_corpus(path, corpus):
    docs = os.listdir(path)
    for doc in tqdm(docs):
        with open(path+doc, 'r') as f:
            doc_text = f.readlines()
        doc_text_parsed = ''.join(doc_text)
        
        corpus.append(Document(page_content=doc_text_parsed, metadata={'source':doc}))
    return corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_model', type=str, default='gte')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    emb_model = args.emb_mode
    device = args.device

    if emb_model == 'bge':
        model_name = 'BAAI/bge-m3'
    elif emb_model == 'gte':
        model_name = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
    elif emb_model == 'm-e5':
        model_name = 'intfloat/multilingual-e5-large-instruct'
    else:
        emb_model == 'gte'
        model_name = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
    
    print("Loading data...")
    path = 'unipa-gpt/' # path till main folder

    dipartimenti = json.load(open(path+'dipartimenti.json'))

    corpus = []
    corpus = load_corpus(path+'corpora/unipa-corpus-'+corpus_type+'-docs/', corpus)

    print("Splitting text...")
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    documents = text_splitter.split_documents(corpus)

    print("Loading embeddings...")

    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        #multi_process=True,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}) # Set `True` for cosine similarity

    print("Creating vector store...")

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)

    os.makedirs(f"{path}embeddings/", exist_ok=True)
    
    vectorstore.save_local(f"{path}embeddings/"+emb_model)

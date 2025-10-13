from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import argparse
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
    parser.add_argument('--corpus_type', type=str, default='clear')
    args = parser.parse_args()
    corpus_type = args.corpus_type

    if corpus_type not in ['full', 'emb', 'clear']:
        corpus_type = 'clear'

    model_name = 'text-embedding-ada-002'

    print("Loading data...")
    path = 'unipa-gpt/' # path till main folder

    corpus = []
    corpus = load_corpus(path+'corpora/unipa-corpus-'+corpus_type+'-docs/', corpus)

    print("Splitting text...")
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    documents = text_splitter.split_documents(corpus)

    print("Loading embeddings...")

    # insert your openai api key
    API_KEY = ""

    embeddings_model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=API_KEY,
    )

    print("Creating vectorstore...")
    
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    
    os.makedirs(f"{path}embeddings/", exist_ok=True)
    
    vectorstore.save_local(f"{path}embeddings/courpus-"+corpus_type)

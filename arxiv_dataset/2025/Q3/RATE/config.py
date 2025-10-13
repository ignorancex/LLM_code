import os

TECH_RAG_JSON_PATH = "./Data/RAG_Knowledge_Base_Example.json" # List of technologies being used as the embeddings of the RAG system, look at the example.

OUTPUT_DIR = "./VectorStore"
EMBEDDING_MODEL_NAME = "bge-m3:latest" # Name of the Ollama embeder model (first install Ollama)
PERSIST_DIRECTORY = os.path.join(OUTPUT_DIR, "chroma")

N_DOC_RETRIEVE = 20 # Primary documents retrieved by the retriever 
TARGET_DIVERSE_DOCS_COUNT = 7 # Selection of secondary documents by diversification
FILTER_CONFIDENCE = 0.7 # Priamary LLM confidence score for extracted technology

LLM_API_KEY = "..." # Provide your own DeepSeek API key
LLM_BASE_URL = "https://api.deepseek.com/v1" 

SPACY_MODEL = "en_core_web_lg" # Name of the SpaCy model that is being used in TechnologyExtractor class
from retrieval_algorithms import  EmbeddingModelBaseClass, Openai_embeddings_v3
import pandas as pd
from pathlib import Path

# Custom modules
from log import get_logger

# Configure logging
logger = get_logger(__name__)

logger.info("Building OpenAI embeddings vector database with model 'text-embedding-3-large'...")
embedding_model = Openai_embeddings_v3(model="text-embedding-3-large")
embedding_model.build_index()

eval_df = pd.read_csv(Path('data/eval_dataset.csv'), index_col=0)

def get_search_results(
        queries: list[str], 
        embedding_model: EmbeddingModelBaseClass,
        top_k: int = 10, 
        ) -> list[str]:
    search_results = embedding_model.retrieve(queries, include_metadatas=True, top_k=top_k)
    return search_results

if 'search_engine_results_top_1' not in eval_df.columns:
    logger.warning("Retrieving top 1 result for evaluation queries from the vector database...")
    openai_large_embedding_model = Openai_embeddings_v3(model="text-embedding-3-large")
    search_results = get_search_results(
        queries=eval_df['query'].tolist(), 
        top_k=1, 
        embedding_model=openai_large_embedding_model)
    eval_df['search_engine_results_top_1'] = search_results
    eval_df.to_csv('data/eval_dataset.csv')

if 'search_engine_results_top_200' not in eval_df.columns:
    logger.warning("Retrieving top 200 results for evaluation queries from the vector database...")
    openai_large_embedding_model = Openai_embeddings_v3(model="text-embedding-3-large")
    search_results = get_search_results(
        queries=eval_df['query'].tolist(), 
        top_k=200, 
        embedding_model=openai_large_embedding_model)
    eval_df['search_engine_results_top_200'] = search_results
    eval_df.to_csv('data/eval_dataset.csv')

logger.warning("All set for evaluation!")
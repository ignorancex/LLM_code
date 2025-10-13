import uuid
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import pandas as pd
from pathlib import Path
from typing import Union
from tqdm import tqdm
from datetime import datetime as dt
from dotenv import load_dotenv
import os

# Custom modules
from log import get_logger

# Configure logging
logger = get_logger(__name__)

# Load relevant API keys
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Load relevant data
data_path = Path("data", "extended_thesaurus.csv")
THESAURUS = pd.read_csv(data_path, index_col=0)
THESAURUS_EXPRESSIONS_LIST = THESAURUS["expression"].to_list()

# List with all the queries to evaluate the algorithms
query_data_path = Path("data","eval_dataset.csv")
query_data_df = pd.read_csv(query_data_path, index_col=0)
QUERY_DATA_LIST = query_data_df["query"].to_list()

# Path to save the results of the retrieval algorithms
queries_results_path = Path("results","queries_results.csv")

# Create relevant directories if they do not exist yet
vector_database_path = Path('vector_database')
if not vector_database_path.exists():
    vector_database_path.mkdir(parents=True, exist_ok=True)


class EmbeddingModelBaseClass():
    def __init__(self):
        raise NotImplementedError
    
    def get_embeddings(self):
        raise NotImplementedError
    
    def build_index(self):
        raise NotImplementedError
    
    def retrieve(self):
        raise NotImplementedError


class Openai_embeddings(EmbeddingModelBaseClass):
    """
    This class handles all processes related to the Openai embedding model \
    and to the information retrieval with it.
    """
    def __init__(self,
                 api_key : str = openai_api_key,
                 model : str = "text-embedding-ada-002",
                 ):
        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "openai_embeddings"

        # Authentication at OpenAI API
        openai.api_key = api_key
        self.openai_client = OpenAI()

    def get_embeddings(self, documents : list[str], return_token_usage : bool = False) -> list[list[float]]:
        if return_token_usage:
            response = self.openai_client.embeddings.create(input = documents, model=self.model)
            return [emb.embedding for emb in response.data], response.usage.total_tokens

        embeddings = self.openai_client.embeddings.create(input = documents, model=self.model).data
        return [emb.embedding for emb in embeddings]    
    
    def add_documents_to_db(self, documents : pd.DataFrame = THESAURUS, batch_size : int = 1000) -> None:
        logger.info(f"Adding documents to collection: {self.db_collection_name}")
        # Get collection
        collection = self.db_client.get_collection(self.db_collection_name)
        # Get documents that already are in the vectorstore
        documents_in_vectorstore = collection.get()['documents']
        # Define pending documents
        pending_documents = [doc for doc in documents.to_dict('records') if doc["expression"] not in documents_in_vectorstore]
        # Add documents with embeddings
        for i in tqdm(range(0, len(pending_documents), batch_size)):
            batch_documents = pending_documents[i:i+batch_size]
            embeddings = self.get_embeddings([doc["expression"] for doc in batch_documents])
            collection.add(ids=[str(uuid.uuid1()) for _ in range(len(batch_documents))], 
                            documents=[doc["expression"] for doc in batch_documents],
                            embeddings=embeddings,
                            metadatas=[{"code": doc["code"], "expression": doc["expression"]} for doc in batch_documents])
        logger.info(f"{len(pending_documents)} added to collection: {self.db_collection_name}")

    def setup_vector_database(self) -> None:
        # make sure that collection exists
        # define cosine similarity as the distance function through Hierarchical Navigable Small Worlds (HNSW)
        logger.info(f"Checking ChromaDB collection: {self.db_collection_name}")
        collection = self.db_client.get_or_create_collection(name=self.db_collection_name,
                                                             metadata={"hnsw:space": "cosine"}) 

        documents_in_vectorstore = collection.get()['documents']

        # check if the documents are exactly what we need
        if sorted(set(documents_in_vectorstore)) != sorted(set(THESAURUS_EXPRESSIONS_LIST)):
            logger.info(f"The data in the collection is not complete or incorrect. Applying corrections to {self.db_collection_name}")
            
            logger.info("Removing documents that should not be in the collection.")
            incorrect_documents = [item for item in documents_in_vectorstore if item not in THESAURUS_EXPRESSIONS_LIST]
            for item in tqdm(incorrect_documents):
                collection.delete(where={"expression": {"$eq": item}})
            logger.info(f"{len(incorrect_documents)} were removed from {self.db_collection_name} collection.")

            logger.info("Checking pending documents.")
            self.add_documents_to_db()

    def build_index(self) -> None:
        self.setup_vector_database()
        logger.info(f"Collection {self.db_collection_name} is ready for information retrieval.")

    def retrieve(self, 
                 input: Union[str, list[str]], 
                 top_k: int = 10, 
                 batch_size : int = 1000,
                 return_token_usage : bool = False,
                 ) -> list[list[str]]:
        """
        This function handles the information retrieval process and returns a list \
        of ICPC codes based on the queries given.

        It applies the desired preprocessing through the arguments remove_special_chars \
        and lowercase.

        input : query for the retrieval
        top_k : number of top results to return
        data : this is the data used to build the BM25 index and will be used to retrieve the results
        batch_size : size of batches to generate embeddings
        """

        if isinstance(input, str):
            input = [input]

        assert isinstance(input, list) and all(isinstance(item, str) for item in input), \
            "Input must be either of type str or list[str]"

        # Get collection
        collection = self.db_client.get_collection(self.db_collection_name)

        logger.info(f"Retrieving with Collection {self.db_collection_name}...")
        
        # Get embeddings in batches
        inputs_embeddings = []
        total_token_usage = 0
        for i in tqdm(range(0, len(input), batch_size), desc=f"Generating query embeddings in batches of {batch_size}"):
            if return_token_usage:
                embeddings, token_usage = self.get_embeddings(input[i:i+batch_size], return_token_usage=return_token_usage)
                inputs_embeddings += embeddings
                total_token_usage += token_usage
            else:
                inputs_embeddings +=self.get_embeddings(input[i:i+batch_size])

        # Query collection
        results = collection.query(query_embeddings=inputs_embeddings, 
                            include=["documents", "metadatas", "distances"], 
                            n_results=top_k)
        
        # Gather results and their metadata
        gathered_results = []
        for metadatas, distances in zip(results["metadatas"], results["distances"]):
            metadatas = [{**metadata, 'distance': -(distance-1)} for metadata, distance in zip(metadatas, distances)]
            gathered_results.append(metadatas)
        
        logger.info("Done!")

        if return_token_usage:
            return gathered_results, total_token_usage

        return gathered_results
        

class Openai_embeddings_v3(Openai_embeddings):
    def __init__(self,
                 api_key : str = openai_api_key,
                 model : str = "text-embedding-3-small",
                 ):
        
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

        if model not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}, but got '{model}'")

        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "openai_embeddings_" + model

        # Authentication at OpenAI API
        openai.api_key = api_key
        self.openai_client = OpenAI()
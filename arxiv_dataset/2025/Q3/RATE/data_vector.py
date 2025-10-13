import os
import logging
from typing import List
from tqdm.auto import tqdm
import shutil

from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config
import doc_processing


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_vectorstore(documents: List[Document]) -> None:

    if not documents:
        raise ValueError("No documents provided for embedding.")

    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
    )

    for document in tqdm(documents):
        vectorstore.add_documents([document])

    logging.info(f"Created DataVector in : {config.PERSIST_DIRECTORY}")


def create_retriever():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory set to: {config.OUTPUT_DIR}")

    if os.path.exists(config.PERSIST_DIRECTORY):
        logging.info(f"Removing existing Chroma directory: {config.PERSIST_DIRECTORY}")
        shutil.rmtree(config.PERSIST_DIRECTORY)

    # Load JSON data
    data = doc_processing.load_json_data(config.TECH_RAG_JSON_PATH)

    # Prepare documents
    documents = doc_processing.prepare_documents(data)

    if documents:
        sample_metadata = documents[0].metadata
        logging.info(f"Sample document metadata keys: {list(sample_metadata.keys())}")
        if 'description' in sample_metadata and 'applications' in sample_metadata:
            logging.info("Metadata contains all required fields.")
        else:
            logging.warning("Metadata is missing some fields!")

    # Split documents
    documents = doc_processing.split_documents(documents)

    if documents:
        sample_split_metadata = documents[0].metadata
        logging.info(f"Sample split metadata keys: {list(sample_split_metadata.keys())}")
        if 'description' in sample_split_metadata and 'applications' in sample_split_metadata:
            logging.info("Split metadata contains all required fields.")
        else:
            logging.warning("Split metadata is missing some fields!")

    # Create datavector
    documents = doc_processing.convert_metadata_lists_to_strings(documents)
    create_vectorstore(documents)

    logging.info("Processing completed successfully!")
    logging.info("Next step: Run your technology extraction pipeline with the updated embeddings.")


def load_retriever():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': config.N_DOC_RETRIEVE})
    return retriever


def retrieve_context(question: str, retriever) -> List[Document]:
    """
    Retrieves relevant documents from the precomputed vector store using the question.
    Applies diversity filtering to the retrieved set.
    Relies ONLY on the provided retriever and the precomputed documents in the vector store.
    """

    # Step 1: Retrieve documents from the precomputed vector store
    # The 'retriever' is assumed to be configured with the precomputed vector store
    documents = retriever.invoke(question)
    logging.debug(f"Retrieved {len(documents)} raw documents for query: '{question[:100]}...'")

    if not documents:
        logging.warning(f"No documents found in the precomputed set for query: '{question[:100]}...'. Returning empty list.")
        return []

    # Step 2: Apply Diversity Filtering to the retrieved documents
    diverse_docs: List[Document] = []
    seen_types: set[str] = set()
    seen_domains: set[str] = set()

    # First pass for diversity
    for doc in documents:
        if not isinstance(doc, Document) or not hasattr(doc, 'metadata'):
            logging.warning(f"Skipping invalid document object during diversity filtering: {type(doc)}")
            continue
        
        doc_type = doc.metadata.get('type', 'UnknownType')
        doc_domain = doc.metadata.get('domain', 'UnknownDomain')
        doc_name = doc.metadata.get('name', 'UnknownName')

        if doc_type not in seen_types or doc_domain not in seen_domains:
            if doc not in diverse_docs: # Ensure document instance uniqueness
                diverse_docs.append(doc)
                seen_types.add(doc_type)
                seen_domains.add(doc_domain)
                logging.debug(f"Added for diversity: '{doc_name}' (Type: {doc_type}, Domain: {doc_domain})")
        
        if len(diverse_docs) >= config.TARGET_DIVERSE_DOCS_COUNT:
            break
    
    # Second pass to fill up if not enough diverse documents were found
    if len(diverse_docs) < config.TARGET_DIVERSE_DOCS_COUNT:
        logging.debug(f"Diversity pass yielded {len(diverse_docs)} docs. Attempting to add more (up to {config.TARGET_DIVERSE_DOCS_COUNT}).")
        for doc in documents: # Iterate through the original retrieved documents again
            if len(diverse_docs) >= config.TARGET_DIVERSE_DOCS_COUNT:
                break
            if doc not in diverse_docs: # Add if not already present
                diverse_docs.append(doc)
                doc_name = doc.metadata.get('name', 'UnknownName') if hasattr(doc, 'metadata') else 'UnknownName'
                logging.debug(f"Added additional doc (filling up): '{doc_name}'")

    if not diverse_docs:
        logging.info(f"No documents selected after diversity filtering for query: '{question[:100]}...'.")
        return [] # Should ideally not happen if initial 'documents' was not empty

    logging.info(f"Selected {len(diverse_docs)} diverse documents for query: '{question[:100]}...'.")
    for i, doc in enumerate(diverse_docs[:config.TARGET_DIVERSE_DOCS_COUNT]): # Log only the ones being returned
        if hasattr(doc, 'metadata'):
            logging.debug(f"Final context doc {i+1}: '{doc.metadata.get('name', 'N/A')}' "
                            f"(Type: {doc.metadata.get('type', 'N/A')}, Domain: {doc.metadata.get('domain', 'N/A')})")
    
    return diverse_docs[:config.TARGET_DIVERSE_DOCS_COUNT] # Return up to the target count
import os
import re
import json
import logging
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_json_data(path: str) -> List[Dict[str, Any]]:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    logging.info(f"Loaded {len(data)} entries from JSON file")
    return data


def preprocess_text(text: str) -> str:
    """Removes extra whitespace from text."""
    return re.sub(r'\s+', ' ', text).strip()


def sanitize(value: Any, default: str = 'N/A') -> str:
    s_val = str(value if value is not None else default)
    # Escape curly braces for f-string compatibility
    return s_val.replace('{', '{{').replace('}', '}}')


def convert_metadata_lists_to_strings(documents: List[Document]) -> List[Document]:
    """Converts list values in metadata records to comma-separated strings."""
    
    for document in documents:
        for key, value in document.metadata.items():
            if isinstance(value, list):
                document.metadata[key] = ", ".join(map(str, value))
                
    return documents


def prepare_documents(data: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for entry in data:
        text = f"""Name: {entry['name']}
Type: {entry['type']}
Domain: {entry['domain']}
Description: {entry['description']}
Applications: {', '.join(entry['applications'])}
Related Technologies: {', '.join(entry['related'])}
Tags: {', '.join(entry['tags'])}"""
        
        metadata = {
            'unique_id': entry['unique_id'],
            'name': entry['name'],
            'type': entry['type'],
            'domain': entry['domain'],
            'description': entry['description'],
            'applications': entry['applications'],
            'related': entry['related'],
            'tags': entry['tags']
        }
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    logging.info(f"Prepared {len(documents)} documents for embedding")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_splits = text_splitter.split_documents(documents)
    
    logging.info(f"Split documents into {len(all_splits)} chunks")
    return all_splits


def load_spacy_model():
    try:
        import spacy
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        print("Downloading spaCy model en_core_web_lg...")
        import spacy.cli
        spacy.cli.download(config.SPACY_MODEL)
        nlp = spacy.load(config.SPACY_MODEL)
    print(f"spaCy model {config.SPACY_MODEL} loaded successfully.")
    return nlp
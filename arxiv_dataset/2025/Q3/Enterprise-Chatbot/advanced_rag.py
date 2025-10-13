import os
import numpy as np
import faiss
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import spacy
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

import json
import random

PDF_FOLDER_PATH = r"AZURE_AI\docs"
INDEX_PATH = r"AZURE_AI\process\faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
user_feedback_log = []

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE})
reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=reranker, top_n=5)
llm = ChatGroq(model="mistral-saba-24b", api_key=os.getenv("GROQ_API_KEY"), temperature=0.0, max_tokens=512)

GROUNDED_PROMPT = PromptTemplate(
    input_variables=["query", "sources"],
    template="""You are an AI assistant answering queries based solely on the provided sources.
# Answer the query using only the facts in the sources below.
# Use bullets for multiple points.
# If the answer exceeds 3 sentences, include a summary.
# Cite the source (title) for each fact.
# If the information is insufficient, say "I don't have enough information.
Query: {query}
Sources:
{sources}"""
)


def prepare_documents():
    nlp = spacy.load("en_core_web_sm")
    loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    document_types = ['HR Policy', 'Financial Report', 'Technical Manual', 'Internal Memo']
    departments = ['HR', 'Finance', 'Engineering', 'Operations']
    confidentiality_levels = ['Public', 'Internal', 'Confidential']

    for i, doc in enumerate(docs):
        spacy_doc = nlp(doc.page_content)
        locations = [ent.text for ent in spacy_doc.ents if ent.label_ == "GPE"]
        doc.metadata["locations"] = locations
        doc.metadata["title"] = doc.metadata.get("source", "unknown").split("/")[-1]
        doc.metadata["parent_id"] = doc.metadata["source"]
        doc.metadata["chunk_id"] = f"{doc.metadata['source']}_chunk_{i}"

        # Enterprise metadata
        doc.metadata["document_type"] = random.choice(document_types)
        doc.metadata["department"] = random.choice(departments)
        doc.metadata["confidentiality_level"] = random.choice(confidentiality_levels)
        doc.metadata["author"] = f"Author_{random.randint(1,10)}"

    return docs


def create_indexes(docs):
    doc_vectors = np.array([embeddings.embed_documents([doc.page_content])[0] for doc in docs], dtype=np.float32)
    dimension = doc_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50
    index.add(doc_vectors)
    faiss.write_index(index, INDEX_PATH)
    vector_store = FAISS.from_documents(docs, embeddings, distance_strategy="L2")
    vector_store.save_local(INDEX_PATH + "_langchain")


def metadata_aware_filter(doc, query_terms):
    if 'locations' in doc.metadata:
        return any(term.lower() in [loc.lower() for loc in doc.metadata['locations']] for term in query_terms)
    return False


# def search_and_answer(query, k=5, query_terms=[]):
#     vector_store = FAISS.load_local(INDEX_PATH + "_langchain", embeddings, allow_dangerous_deserialization=True)
#     bm25_retriever = BM25Retriever.from_documents(prepare_documents(), k=10)
#     vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

#     bm25_results = bm25_retriever.get_relevant_documents(query)
#     vector_results = vector_retriever.get_relevant_documents(query)

#     retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=vector_store.as_retriever(search_kwargs={"k": 10})
#     )
#     reranked_results = retriever.get_relevant_documents(query)

#     print("\n--- Retrieved Chunks with Scores and Metadata ---")
#     for idx, doc in enumerate(reranked_results):
#         print(f"Rank {idx+1} | Title: {doc.metadata.get('title', 'unknown')} | Score: {doc.metadata.get('score', 'N/A')}")
#         print(f"Type: {doc.metadata.get('document_type')}, Department: {doc.metadata.get('department')}, Confidentiality: {doc.metadata.get('confidentiality_level')}, Author: {doc.metadata.get('author')}")
#         print(f"Content Preview: {doc.page_content[:200]}\n")

#     filtered_results = [doc for doc in reranked_results if metadata_aware_filter(doc, query_terms)]
#     results = filtered_results if filtered_results else reranked_results[:k]

#     sources_formatted = "\n=================\n".join(
#         [f'TITLE: {doc.metadata.get("title", "unknown")}, CONTENT: {doc.page_content}' for doc in results]
#     )

#     prompt = GROUNDED_PROMPT.format(query=query, sources=sources_formatted)
#     response = llm.invoke(prompt)

#     memory.save_context({"user_input": query}, {"response": response.content})

#     return response.content

def search_and_answer(query, k=5, query_terms=[]):
    vector_store = FAISS.load_local(INDEX_PATH + "_langchain", embeddings, allow_dangerous_deserialization=True)
    bm25_retriever = BM25Retriever.from_documents(prepare_documents(), k=10)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 10})
    )
    reranked_results = retriever.get_relevant_documents(query)

    retrieved_chunks = []
    print("\n--- Retrieved Chunks with Metadata ---")
    for idx, doc in enumerate(reranked_results):
        # Dummy score assignment (you can replace this with a real similarity or reranker score if available)
        score = doc.metadata.get('score', 0.5)  # Default dummy score

        print(f"Rank {idx+1} | Title: {doc.metadata.get('title', 'unknown')} | Score: {score}")
        print(f"Content Preview: {doc.page_content[:200]}\n")

        retrieved_chunks.append({
            "score": score,
            "metadata": doc.metadata,
            "content": doc.page_content
        })

    filtered_results = [doc for doc in reranked_results if metadata_aware_filter(doc, query_terms)]
    final_results = filtered_results if filtered_results else reranked_results[:k]

    sources_formatted = "\n=================\n".join(
        [f'TITLE: {doc.metadata.get("title", "unknown")}, CONTENT: {doc.page_content}' for doc in final_results]
    )

    prompt = GROUNDED_PROMPT.format(query=query, sources=sources_formatted)
    response = llm.invoke(prompt)

    memory.save_context({"user_input": query}, {"response": response.content})

    return response.content, retrieved_chunks


def expand_query_with_llm(query):
    prompt = f"Refine or expand the following user query to retrieve more relevant information:\n\nQuery: {query}\n\nRefined Query:"
    expanded = llm.invoke(prompt)
    return expanded.content.strip()


def log_user_feedback(query, response, feedback, query_terms=[]):
    user_feedback_log.append({"query": query, "response": response, "feedback": feedback})
    with open("user_feedback.json", "w") as f:
        json.dump(user_feedback_log, f, indent=4)

    if feedback.lower() == "no":
        print("\nUser was not satisfied. Expanding the query and retrying with more documents...")

        expanded_query = expand_query_with_llm(query)
        print(f"Expanded Query: {expanded_query}")

        refined_response, _ = search_and_answer(expanded_query, k=10, query_terms=query_terms)
        print("\nImproved Response after Query Refinement:\n")
        print(refined_response)

        return refined_response  # Return refined response
    return response  # Return original if feedback is 'yes'



if __name__ == "__main__":
    print("Preparing and indexing documents...")
    documents = prepare_documents()
    create_indexes(documents)
    print("Done.")

    query = "tell me about chetak ev?"
    response = search_and_answer(query, k=5, query_terms=["water", "quality"])
    print("\nInitial Response:")
    print(response)

    feedback = input("\nWas this answer helpful? (yes/no): ")
    log_user_feedback(query, response, feedback, query_terms=["water", "quality"])

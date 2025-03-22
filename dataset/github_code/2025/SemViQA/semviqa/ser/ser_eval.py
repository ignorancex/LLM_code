import numpy as np
import torch
import torch.nn as nn
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from ..data_processing.pipline import split_sentence, preprocess_text


def extract_evidence_qatc(claim, context, model_evidence_QA, tokenizer_QA, device):
    """
    Extracts evidence from the given context using a QA model.
    
    Args:
        claim (str): The claim statement.
        context (str): The context containing possible evidence.
        model_evidence_QA (torch.nn.Module): The QA model used for evidence extraction.
        tokenizer_QA: Tokenizer corresponding to the QA model.
        device (str): The device (CPU/GPU) to run the model on.

    Returns:
        str or int: The extracted evidence sentence or -1 if no valid evidence is found.
    """
    model_evidence_QA.to(device).eval()
    inputs = tokenizer_QA(claim, context, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_evidence_QA(**inputs)

    start_index, end_index = np.argmax(outputs.start_logits.cpu().numpy()), np.argmax(outputs.end_logits.cpu().numpy())
    evidence = tokenizer_QA.decode(inputs['input_ids'][0][start_index:end_index + 1]).replace('<s>', '').replace('</s>', '')

    if not evidence or len(split_sentence(evidence)) > 1:
        return -1

    for line in split_sentence(context):
        if preprocess_text(evidence) in preprocess_text(line):
            return line

    print('Error: Evidence not found in context')
    return evidence


def extract_evidence_qatc_faster(claim, context, full_context, model_evidence_QA, tokenizer_QA, device):
    """
    Faster evidence extraction method using batch processing.

    Args:
        claim (str or list): The claim(s) to be processed.
        context (str or list): The corresponding context(s).
        full_context (str): The full original context.
        model_evidence_QA (torch.nn.Module): The QA model used for evidence extraction.
        tokenizer_QA: Tokenizer corresponding to the QA model.
        device (str): The device (CPU/GPU) to run the model on.

    Returns:
        str or int: The extracted evidence or -1 if no valid evidence is found.
    """
    claim, context = ([claim], [context]) if isinstance(claim, str) else (claim, context)
    
    model_evidence_QA.to(device).eval()
    inputs = tokenizer_QA(claim, context, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model_evidence_QA(**inputs)

    start_indices = outputs.start_logits.argmax(dim=-1).tolist()
    end_indices = outputs.end_logits.argmax(dim=-1).tolist()

    evidences = tokenizer_QA.batch_decode(
        [inputs['input_ids'][0][s:e + 1] for s, e in zip(start_indices, end_indices)],
        skip_special_tokens=True
    )

    valid_evidence = [(preprocess_text(evi.lstrip(".")), i) for i, evi in enumerate(evidences) if len(preprocess_text(evi).split()) > 3]

    if len(valid_evidence) != 1:
        return -1  

    evidence_text = valid_evidence[0][0]
    for line in split_sentence(full_context):
        if evidence_text in preprocess_text(line):
            return line  

    return -1  


def tfidf_topk(claim, context, threshold=0.6, top_k=1):
    """
    Retrieves the top-k relevant sentences from the context using TF-IDF similarity.

    Args:
        claim (str): The claim statement.
        context (str): The context containing multiple sentences.
        threshold (float): Threshold for sentence length ratio.
        top_k (int): Number of top sentences to return.

    Returns:
        list: List of tuples (similarity_score, sentence).
    """
    corpus = split_sentence(context)
    processed_claim = preprocess_text(ViTokenizer.tokenize(claim).lower())
    
    corpus_processed = []
    for i, sentence in enumerate(corpus):
        sentence = preprocess_text(ViTokenizer.tokenize(sentence).lower())
        if i > 0 and 1 < len(sentence.split()) / len(processed_claim.split()) < threshold:
            sentence = f"{corpus[i-1]}. {sentence}"
        corpus_processed.append(sentence)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_processed + [processed_claim])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    return sorted(zip(cosine_sim, corpus), reverse=True)[:top_k]


def bm25_topk(claim, context, top_k=None):
    """
    Retrieves the top-k relevant sentences from the context using BM25 ranking.

    Args:
        claim (str): The claim statement.
        context (str): The context containing multiple sentences.
        top_k (int, optional): Number of top sentences to return. If None, returns all.

    Returns:
        list: List of tuples (score, sentence).
    """
    sentences = split_sentence(context)
    tokenized_context = [sentence.split(' ') for sentence in sentences]
    bm25 = BM25Okapi(tokenized_context)
    scores = bm25.get_scores(claim.split())

    normalized_scores = MinMaxScaler().fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    sorted_sentences = sorted(zip(normalized_scores, sentences), reverse=True)

    return sorted_sentences[:top_k] if top_k else sorted_sentences


def mean_pooling(model_output, attention_mask):
    """
    Computes mean pooling of token embeddings for sentence representation.

    Args:
        model_output (torch.Tensor): Output embeddings from a transformer model.
        attention_mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: Mean-pooled sentence representation.
    """
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def extract_evidence_tfidf_qatc(claim, context, model_evidence_QA, tokenizer_QA, device, confidence_threshold=0.5, length_ratio_threshold=0.6, is_qatc_faster=False):
    """
    Finds the most relevant evidence for a given claim in the context.

    Args:
        claim (str): The claim statement.
        context (str): The context containing possible evidence.
        model_evidence_QA (torch.nn.Module): The QA model used for evidence extraction.
        tokenizer_QA: Tokenizer corresponding to the QA model.
        device (str): The device (CPU/GPU) to run the model on.
        confidence_threshold (float): Similarity threshold for TF-IDF selection.
        is_qatc_faster (bool): Whether to use the faster QATC method.

    Returns:
        str: The best-matching evidence sentence.
    """
    evidence_tf = tfidf_topk(claim, context, top_k=1, threshold=length_ratio_threshold)[0]
    if evidence_tf[0] > confidence_threshold:
        return evidence_tf[1]

    sentences = split_sentence(context)
    if len(context.split()) <= 400:  
        evidence = extract_evidence_qatc(claim, context, model_evidence_QA, tokenizer_QA, device)
        return evidence if evidence != -1 else evidence_tf[1]

    token_sentences = [l.split(' ') for l in sentences]

    tmp_context_token = []
    tmp_context = []
    sub_contexts = []
    for idx in range(len(sentences)):
        check = True
        if len(token_sentences[idx] + tmp_context_token) <=400:
            tmp_context_token += token_sentences[idx]
            tmp_context.append(sentences[idx])
            check = False
        
        if len(token_sentences[idx] + tmp_context_token) > 400 or idx == len(sentences) - 1:
            context_sub = '. '.join(tmp_context)
            if len(context_sub)== 0: 
                continue
            sub_contexts.append(context_sub)

            if check:
                tmp_context_token = token_sentences[idx]
                tmp_context = [sentences[idx]]
            else:
                tmp_context_token = []
                tmp_context = []

    if is_qatc_faster:
        return extract_evidence_qatc_faster(claim, sub_contexts, context, model_evidence_QA, tokenizer_QA, device)

    for sub_context in sub_contexts:
        evidence = extract_evidence_qatc(claim, sub_context, model_evidence_QA, tokenizer_QA, device)
        if evidence != -1:
            return evidence

    return evidence_tf[1]

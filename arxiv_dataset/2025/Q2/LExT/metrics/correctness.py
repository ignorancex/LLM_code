import pandas as pd
import torch
import logging
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from src.basic_functions import call_llama
from src.utils import save_to_references

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set transformers logging to error only
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


def get_bert_embedding(text):
    if pd.isna(text):
        return None
    elif not isinstance(text, str):
        text = str(text)
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():   
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

def compute_cosine_similarity_bert(text1, text2):
    if text1 is None or text2 is None:
        return 0
    vec1 = get_bert_embedding(text1)
    vec2 = get_bert_embedding(text2)
    return cosine_similarity([vec1], [vec2])[0][0]

def weighted_accuracy(ground_truth, predicted_exp, ner_pipe=None, row_reference={}, save=True):
    # cosine similarity between ground truth and predicted explanation
    base_acc = compute_cosine_similarity_bert(ground_truth, predicted_exp)
    print("Computing Weighted Accuracy\n")

    if not ner_pipe:
        ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

    # NER-based overlap
    def extract_words_from_ner(text):
        entities = ner_pipe(text)
        words = [entity['word'] for entity in entities]
        return set(words)
    
    ground_truth_words = extract_words_from_ner(ground_truth)
    predicted_words = extract_words_from_ner(predicted_exp)
    
    if predicted_words:
        overlap_fraction = len(ground_truth_words & predicted_words) / len(predicted_words)
    else:
        overlap_fraction = 1e-8
    
    final_accuracy = (overlap_fraction ** 0.2) * base_acc
    
    if save:
        row_reference['predicted_explanation'] = predicted_exp
        row_reference['accuracy'] = final_accuracy
        # save_to_references(row_reference)
    
    return final_accuracy

def context_relevancy(predicted_explanation, ground_question, groq, row_reference={}):

    print("Computing Context Relevancy\n")

    # Generate a new question from the predicted explanation using the larger Llama model
    prompt = f"Generate a question that can be completely answered from the below explanation. Just give me the question. Don't add anything else to your response:\n\nExplanation: {predicted_explanation}"
    generated_question = call_llama(prompt, groq).strip()
    
    
    # Compute cosine similarity between the generated question and the ground question
    context_relevancy_score = compute_cosine_similarity_bert(generated_question, ground_question)
    
    # Save the new question in references
    row_reference['generated_question'] = generated_question
    row_reference['context_relevancy'] = context_relevancy_score
    # save_to_references(row_reference)
    
    return context_relevancy_score

def correctness(ground_explanation, predicted_explanation, ground_question, groq, ner_pipe=None, row_reference={}):

    
    """
    Compute the final correctness as the average of final accuracy and context relevancy.
    """
    accuracy_score = weighted_accuracy(ground_explanation, predicted_explanation, ner_pipe, row_reference)
    context_relevancy_score = context_relevancy(predicted_explanation, ground_question, groq, row_reference)
    print("Computing Correctness\n")
    
    correctness_score = (accuracy_score + context_relevancy_score) / 2.0
    
    # Print and save results
    print(f"Accuracy: {accuracy_score}, Context Relevancy: {context_relevancy_score}, Correctness: {correctness_score}\n")
    row_reference['final_accuracy'] = accuracy_score
    row_reference['final_context_relevancy'] = context_relevancy_score
    row_reference['correctness'] = correctness_score
    # save_to_references(row_reference)
    
    return accuracy_score, context_relevancy_score, correctness_score

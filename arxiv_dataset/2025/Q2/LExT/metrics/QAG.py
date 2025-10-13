import re
from src.basic_functions import call_model, call_llama
import numpy as np

def generate_questions(explanation, groq):
    prompt = (f"Generate at least 5 questions that can be answered using the following explanation. "
              f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
              f"Explanation: {explanation}")
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]

def evaluate_questions(questions, explanation, target_model, row_reference ={}):
    yes_count = 0
    no_i_dont_know_questions = []
    
    for question in questions:
        prompt = (f"Can the following question be answered from this explanation?\n\n"
                  f"Explanation: {explanation}\nQuestion: {question}"
                  f"Just give me a yes/no. Don't add anything else to your answer.")
        answer = call_model(prompt, target_model).strip().lower()
        if "yes" in answer:
            yes_count += 1
        else:
            no_i_dont_know_questions.append(question)
    
    total_questions = len(questions)
    qag_score = yes_count / total_questions if total_questions > 0 else 0
    
    row_reference['qag_yes_count'] = yes_count
    row_reference['qag_total'] = total_questions
    row_reference['qag_score'] = qag_score
    row_reference['qag_failed_questions'] = no_i_dont_know_questions
    
    return qag_score, no_i_dont_know_questions

def qag(explanation, groq, target_model, row_reference={}):
    """
    Compute the QAG score by generating questions from the explanation and evaluating them. "
    """
    print("Computing QAG score\n")

    questions = generate_questions(explanation, groq)
    score, failed = evaluate_questions(questions, explanation, target_model, row_reference)
    return score


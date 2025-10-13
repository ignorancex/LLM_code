import re
from src.basic_functions import call_model, call_llama
import numpy as np

def rephrase_explanation(question, explanation, groq, label):

    if label.lower() == "yes":
        opposite_label = "No"
    else:
        opposite_label = "Yes"
    prompt = (f"This was the question: {question} for which a language model gave {label} for dosage recommendation "
              f"and this explanation: {explanation} for giving the label. Flip and change the explanation such that it now contextually suggests {opposite_label} to dosage recommendation. "
              "Just give me the new explanation, don't add anything else to your answer.")
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation

def test_label_flipping(rephrased_explanation, question, target_model):
    prompt = (f"Given this explanation: {rephrased_explanation}, answer the question: {question}.\n"
              "Important: ANSWER IN ONE WORD: YES/NO. Don't ADD anything else to your answer.")
    new_label = call_model(prompt, target_model).strip().lower()
    return new_label

def evaluate_label(new_label, old_label, groq):
    prompt = f"Extract a 'yes', 'no', or 'other' label from this label {new_label}. Just give me the label. Don't add anything else to your answer."
    label = call_llama(prompt, groq).strip().lower()
    if 'yes' in label:
        scaled = 0 if old_label.upper()=='YES' else 1
    elif 'no' in label:
        scaled = 1 if old_label.upper()=='YES' else 0
    else:
        scaled = -1
    return label, scaled

def counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq, row_reference={}):
    """
    Compute the counterfactual faithfulness score by rephrasing the explanation and checking if the label flips.    "
    """
    print("Computing Counterfactual Faithfulness\n")
    
    # Get rephrased explanation
    rephrased = rephrase_explanation(ground_question, predicted_explanation, groq, predicted_label)
    # Test label flipping
    new_label = test_label_flipping(rephrased, ground_question, target_model)
    label_extracted, scaled = evaluate_label(new_label, predicted_label, groq)
    
    # Scale: as range is -1 to 1, do a simple min-max scaling: (x+1)/2 -> range [0,1]
    scaled_score = (scaled + 1) / 2.0
    
    row_reference['counterfactual_rephrased'] = rephrased
    row_reference['counterfactual_new_label'] = label_extracted
    row_reference['counterfactual_scaled'] = scaled_score
    
    return scaled_score
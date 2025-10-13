from metrics.plausibility import plausibility
from metrics.faithfulness import faithfulness
from src.utils import save_to_references
import numpy as np

def harmonic_mean(a, b):
    if a + b == 0:
        return 0
    return 2 * (a * b) / (a + b)

def lext(ground_context, ground_question, ground_explanation, ground_label, target_model, groq, ner_pipe, row_reference ={}):
    """
    Entrance function that computes the trustworthiness score:
    - Computes plausibility (which in turn computes correctness and consistency)
    - Computes faithfulness (from QAG, counterfactual, and contextual experiments)
    - Then computes the trustworthiness score.
    """

    if row_reference=={}:
        # Row reference dictionary for storing all outputs
        row_reference = {
            "ground_context": ground_context,
            "ground_question": ground_question,
            "ground_explanation": ground_explanation,
            "ground_label": ground_label
        }

    # Compute plausibility

    plaus = plausibility(ground_context, ground_question, ground_explanation, target_model, groq, ner_pipe, row_reference)

    # For faithfulness, use the predicted explanation obtained from plausibility function.
    predicted_explanation = row_reference.get('predicted_explanation', '')
    predicted_label = row_reference.get('predicted_label', '')
    faith = faithfulness(predicted_explanation, predicted_label, ground_question, ground_label, ground_context, groq, target_model, row_reference)
    
    # Compute Trustworthiness as the harmonic mean of plausibility and faithfulness
    trustworthiness = harmonic_mean(plaus, faith)
    print(f"LExT (Language Explanation Trustworthiness) Score: {trustworthiness}")
    
    row_reference['trustworthiness'] = trustworthiness
    save_to_references(row_reference)
    print("All model outputs are saved in data/references.csv")
    return trustworthiness


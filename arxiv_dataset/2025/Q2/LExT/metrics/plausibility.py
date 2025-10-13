from metrics.correctness import correctness
from metrics.consistency import consistency
from src.utils import save_to_references
from src.basic_functions import get_prediction

def plausibility(ground_context, ground_question, ground_explanation, target_model, groq, ner_pipe=None, row_reference= None):
    """
    Plausibility is the average of correctness and consistency.
    It calls the basic prediction function to get the predicted explanation,
    then computes correctness and consistency metrics.
    """
    
    # Get the prediction
    if row_reference=={}:
        label, predicted_explanation = get_prediction(ground_context, ground_question, target_model)
        row_reference['predicted_label'] = label
        row_reference['predicted_explanation'] = predicted_explanation
    else:
        predicted_explanation = row_reference.get('predicted_explanation', "")

    # Compute Correctness
    acc, ctx_rel, corr = correctness(ground_explanation, predicted_explanation, ground_question, groq, ner_pipe, row_reference)
    # Compute Consistency
    iter_stab, para_stab, cons = consistency(ground_question, ground_context, target_model, ground_explanation, groq, ner_pipe, row_reference)

    print("Computing plausibility\n")
    plausibility_score = (corr + cons) / 2.0
    print(f"Plausibility: {plausibility_score}\n")
    row_reference['plausibility'] = plausibility_score
    
    return plausibility_score

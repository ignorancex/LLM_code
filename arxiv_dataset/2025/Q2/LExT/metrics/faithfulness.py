from metrics.QAG import qag
from metrics.contextual import contextual_faithfulness
from metrics.counterfactual import counterfactual_faithfulness

def faithfulness(predicted_explanation, predicted_label, ground_question, ground_label, context, groq, target_model, row_reference={}):
    """
    Compute faithfulness as the average of:
      - QAG Score
      - Counterfactual Faithfulness
      - Contextual Faithfulness
    """
    
    qag_score = qag(predicted_explanation, groq, target_model, row_reference)
    counter = counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq, row_reference)
    contextual = contextual_faithfulness(context, ground_question, predicted_label, target_model, groq, row_reference)
    print("Computing Faithfulness\n")
    faithfulness_score = (qag_score + counter + contextual) / 3.0
    print(f"QAG: {qag_score}, Counterfactual: {counter}, Contextual: {contextual}, Faithfulness: {faithfulness_score}\n")
    
    row_reference['faithfulness'] = faithfulness_score
    
    return faithfulness_score
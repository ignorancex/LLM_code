from src.basic_functions import call_model, call_llama
from src.basic_functions import get_prediction
from src.utils import save_to_references
import numpy as np
from metrics.correctness import weighted_accuracy

def iterative_stability(ground_question, context, target_model, ground_truth, groq, ner_pipe, row_reference={}, iterations=5):
    """
    Run the prediction function for a number of iterations and compute the variance
    between the original prediction and subsequent iterations.
    """
    print("Computing Iterative Stability\n")
    predictions = []

    
    # Run for additional iterations
    for i in range(iterations):
        _, exp_iter = get_prediction(context, ground_question, target_model, groq)
        predictions.append(exp_iter)
    
    accuracy_scores = [
        weighted_accuracy(ground_truth, exp, ner_pipe, save=False) for exp in predictions
    ]
    variance = np.var(accuracy_scores)
    stability_score = 1 - variance
    
    # Save iterations in references
    row_reference['iterative_explanations'] = predictions
    row_reference['iterative_stability'] = stability_score
    
    return stability_score

def paraphrase_stability(ground_question, context, target_model, ground_truth, groq, ner_pipe, row_reference= {}):
    """
    Generate paraphrased versions of the ground question using the larger Llama model.
    Then run the prediction function on each and compute the variance compared to the original.
    """
    print("Computing Paraphrase Stability\n")
    # Generate 3 paraphrased versions
    paraphrase_prompt = (f"Paraphrase the following question in three different ways:\n"
                         f"Question: {ground_question}\n Don't modify the meaning, just use paraphrased context. Just give me three questions seperated by a newline. Don't add anything else.")
    paraphrased_output = call_llama(paraphrase_prompt, groq)
    paraphrased_versions = [q.strip() for q in paraphrased_output.strip().split('\n') if q.strip()]
    
    # Get original prediction explanation
    _, exp_orig = get_prediction(context, ground_question, target_model, groq)
    
    explanations = []
    for para in paraphrased_versions:
        _, exp_para = get_prediction(context, para, target_model, groq)
        explanations.append(exp_para)

    # Save paraphrases and explanations
    row_reference['paraphrased_questions'] = paraphrased_versions
    row_reference['paraphrased_explanations'] = explanations
    
    explanations.append(exp_orig)
    
    accuracy_scores = [
        weighted_accuracy(ground_truth, exp, ner_pipe, save=False) for exp in explanations
    ]
    variance = np.var(accuracy_scores)
    stability_score = 1 - variance
    

    row_reference['paraphrase_variance'] = stability_score
    
    return stability_score

def consistency(ground_question, context, target_model, ground_truth, groq, ner_pipe, row_reference={}):
    """
    Consistency is the average of iterative stability and paraphrase stability.
    """
    
    iter_stab = iterative_stability(ground_question, context, target_model, ground_truth, groq, ner_pipe, row_reference)
    para_stab = paraphrase_stability(ground_question, context, target_model, ground_truth, groq, ner_pipe, row_reference)
    print("Computing Consistency\n")
    
    consistency_score = (iter_stab + para_stab) / 2.0
    
    print(f"Iterative Stability: {iter_stab}, Paraphrase Stability: {para_stab}, Consistency: {consistency_score}\n")
    row_reference['iterative_stability'] = iter_stab
    row_reference['paraphrase_stability'] = para_stab
    row_reference['consistency'] = consistency_score
    return iter_stab, para_stab, consistency_score

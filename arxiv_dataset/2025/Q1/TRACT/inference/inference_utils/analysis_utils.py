from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import logging
from colorama import Fore, Style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_entropy(prob_lists: List):
    """
    Calculate the entropy of a list of probabilities.
    """
    average_probs = np.mean(prob_lists, axis=0)
    logger.info(f"Average probabilities: {Fore.GREEN}{average_probs}{Style.RESET_ALL}")
    probs = np.array(prob_lists)
    # Deal with probabilities that are 0
    probs = np.where(probs == 0, 1e-10, probs)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    return entropy


def calculate_cot_rubric_overlap(
    cots: List[str], 
    rubrics: List[str]
) -> Dict[str, float]:
    """
    Calculate the overlap between the COT (predictions) and rubric (ground truths).
    Possible metrics: ROUGE and BLEU
    
    Args:
        cots (List[str]): List of predicted explanations (COTs).
        rubrics (List[str]): List of reference rubrics (ground truths).
    
    Returns:
        Dict[str, float]: Dictionary containing BLEU and ROUGE-L scores.
    """
    if not cots or not rubrics:
        raise ValueError("Both COTs and rubrics must be non-empty lists.")
    
    # BLEU Calculation
    smoothie = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([r.split()], c.split(), smoothing_function=smoothie) for c, r in zip(cots, rubrics)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # ROUGE Calculation
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(r, c)['rougeL'].fmeasure for c, r in zip(cots, rubrics)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    logger.info(f"{Fore.GREEN}BLEU{Style.RESET_ALL}: {avg_bleu}")
    logger.info(f"{Fore.GREEN}ROUGE-L{Style.RESET_ALL}: {avg_rouge}")

    return {"BLEU": avg_bleu, "ROUGE-L": avg_rouge}

def check_if_cot_contains_many_number(cots: List[str]) -> float:
    """
    Check if the COTs contain many numbers.
    """
    if not cots:
        raise ValueError("The COTs list must be non-empty.")
    
    # Check if the COTs contain many numbers
    num_count = []
    for cot in cots:
        number_in_cot = False
        for word in cot.split():
            if word.isdigit():
                number_in_cot = True
                break
        num_count.append(number_in_cot)                
            
    num_count = np.mean(np.asarray(num_count).astype(float))

    return num_count
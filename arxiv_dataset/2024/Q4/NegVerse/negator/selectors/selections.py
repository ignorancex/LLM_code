from typing import List, Tuple, Dict
import edit_distance
import random
from .detect_neg_cue import filter_neg_sentences

def select_sentences(sentences: List[str], original:str, n_sample: int = 3,lev_thr=0.5) -> List[str]:
    """
    Select a specified number of sentences from a list.

    Parameters:
        sentences (List[str]): A list of sentences to select from.
        n_sample (int): The number of sentences to select. Defaults to 3.
        original (str): input sentence

    Returns:
        List[str]: A list containing the selected sentences.
    """
    
    normalize = lambda s: s.strip().lower().rstrip('.')
    is_valid = True
    validated_set = []

    # Normalize original sentence and create a mapping from normalized to original sentences
    normalized_original = normalize(original)
    sentence_map = {normalize(sentence): sentence for sentence in sentences}

    # Normalize sentences and remove duplicates and the original sentence from the set
    normalized_sentences = set(sentence_map.keys())
    unique_normalized_sentences = normalized_sentences - {normalized_original}

    for norm_sentence in unique_normalized_sentences:
        if "empty" in norm_sentence:
            continue  # Skip the sentence if it contains ['EMPTY']
        
        original_sentence = sentence_map[norm_sentence]
        distance = compute_lev_dist(normalized_original, norm_sentence)
        #print(f"Sentence: {original_sentence}, Levenshtein Distance: {distance}")
        is_valid = distance < lev_thr
        
        if is_valid:
            validated_set.append(original_sentence)

    filtered_sentences = filter_neg_sentences(validated_set)  # Extract sentences containing negation cues
    sampled_sentences = random.sample(filtered_sentences, n_sample) if len(filtered_sentences) > n_sample else filtered_sentences


    return sampled_sentences

def compute_lev_dist(original, generated):

    sm = edit_distance.SequenceMatcher(
        a=original, b=generated)
    reference_length = max(len(original), len(generated))
    distance = sm.distance()
    normalized_distance = distance / reference_length if reference_length > 0 else 0.0
    return normalized_distance

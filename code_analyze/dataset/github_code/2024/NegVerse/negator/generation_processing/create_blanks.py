import numpy as np
from ..helpers import unify_tags, flatten_fillins
from .helpers import Special_tokens
import random
#from ..PEFT.data_preprocess import Special_tokens

def create_blanked_sents(doc, indexes=None):
    if indexes:
        if type(indexes[0]) == int: 
            indexes = [indexes]
        indexes_list = indexes #[indexes]
    else:
        indexes_list = get_random_idxes(
            doc, is_token_only=False, max_count=3)
    blanks = set([flatten_fillins(
        doc, indexes, [Special_tokens.BLANK_TOK] * len(indexes)) \
        for indexes in indexes_list])
    return blanks

def find_matching_indices(doc, pos_list,  dep_list):
    matching_indices = []

    if pos_list is not None:
        for i, token in enumerate(doc):
            #print(pos_list, dep_list)
            if dep_list is None:
                # If dep_list is None, accept any dependency for the given POS
                if token.pos_ in pos_list:
                    matching_indices.append(i)
            else:
                if token.pos_ in pos_list and token.dep_ in dep_list:
                    matching_indices.append(i)

    return matching_indices

# the function for placing BLANKS.
def get_one_random_idx_set(
    doc, max_blank_block=3, req_dep=None, blank_type_prob=None, 
    pre_selected_idxes=None, is_token_only=False):
    if req_dep is not None:
        if type(req_dep) == str: req_dep = [req_dep]
        idx_range_sp = unify_tags(doc,*req_dep) 
        idx_range = idx_range_sp.copy()
        idx_range.extend(find_matching_indices(doc, *req_dep))
    else:
        idx_range = list(range(len(doc)))
    # only keep those pre_selected_idxes
    if pre_selected_idxes is not None:
        idx_range = [i for i in idx_range if i in pre_selected_idxes]
    max_blank_block = min(len(idx_range), max_blank_block)        
    #print(req_dep, idx_range)
    selected_indexes = []

    while max_blank_block > 0 and not selected_indexes:
        # if fixed the thing to change, then do one specific change
        n_perturb = np.random.choice(list(range(1, max_blank_block+1))) #if req_dep is None else 1
        replace_idx, total_run = -1, 1000
        while (total_run > 0 and n_perturb > 0): #and  len(span_and_edits) == 0:
            replace_idx = np.random.choice(idx_range)
            mask = replace_idx in idx_range_sp
            token = doc[replace_idx]
            if token.is_punct:
                total_run -= 1
                continue
            if blank_type_prob: p = blank_type_prob
            else:
                # if fixed the tree, then mostly use the tree
                if is_token_only:  p = [1, 0, 0]
                elif req_dep is None: p = [0.4, 0.35, 0.25]
                else: p = [0.2, 0.8, 0]
            is_replace_subtree = np.random.choice(["token", "subtree", "insert"], p=p)
            if ((is_replace_subtree == "subtree")  or  mask):
                start, end = token.left_edge.i, token.right_edge.i+1
            elif is_replace_subtree == "token" and token.pos_ == 'DET':
                start, end = token.i, token.i+2
            elif is_replace_subtree == "token" and token.pos_ == 'ADJ' and doc[token.i-1].pos_ == 'DET':
                start, end = token.i-1, token.i+2
            elif is_replace_subtree == "token" :
                start, end = token.i, token.i+1
            else:
                start, end = token.i, token.i 
            if all([end < sstart or start > send for sstart, send in selected_indexes]):
                selected_indexes.append([start, end])
                n_perturb -= 1
            total_run -= 1
    return sorted(selected_indexes, key=lambda idx: (idx[0], idx[1]))


def get_random_idxes(doc, 
    pre_selected_idxes=None, 
    deps=None, is_token_only=False, 
    max_blank_block=3, max_count=None):
    unique_blanks = {str([[0, len(doc)]]): [[0, len(doc)]]}
    #default_deps = [None, "", ["subj","obj"], ["aux", "ROOT"], ["conj", "modifier", "clause"]]

    if is_token_only:
        default_deps = [ None, ["subj","obj"],["advmod"],None,["det"]]
        default_pos = [ ["VERB","AUX"], None,["ADV"],["ADJ"],["DET"]]
    else:
        default_deps = [ None, ["subj","obj"],["advmod"],None,["prep"],["det"]]
        default_pos = [ ["VERB","AUX"], None,["ADV"],["ADJ"],["ADP"],["DET"]]

    default_rules = list(zip(default_pos, default_deps))

    if is_token_only: 
        unique_blanks = {}
    if deps is None: deps = default_rules
    for dep in deps:
        rounds = 5
        for _ in range(rounds):
            curr_idx = get_one_random_idx_set(
                doc, req_dep=dep, 
                max_blank_block=max_blank_block,
                pre_selected_idxes=pre_selected_idxes, 
                is_token_only=is_token_only) if dep != "" else None
            if curr_idx is not None and len(curr_idx) > 0:
                unique_blanks[str(curr_idx)] = curr_idx
    unique_blanks = list(unique_blanks.values())
    if max_count is not None:
        try:
            unique_blanks = sample_from_list(unique_blanks, max_count, front_ratio =0.80)

        except:
            unique_blanks = unique_blanks[:max_count]
    return unique_blanks

def sample_from_list(unique_blanks, max_count, front_ratio=0.7):
    """
    Sample items from a list with a preference for the beginning and random sampling from the end.

    Parameters:
    - unique_blanks (list): The list of items to sample from.
    - max_count (int): The maximum number of items to sample.
    - front_ratio (float): The proportion of items to sample from the beginning of the list (default is 0.7).

    Returns:
    - list: A list of sampled items.
    """
    # Ensure unique_blanks is a list
    unique_blanks = list(unique_blanks)
    
    # Number of items to sample from the front and the back
    num_front_samples = int(max_count * front_ratio)
    num_back_samples = max_count - num_front_samples
    
    # Sample from the front of the list
    front_samples = unique_blanks[:min(len(unique_blanks), num_front_samples)]
    
    # Sample randomly from the back of the list
    if num_back_samples > 0 and len(unique_blanks) > num_front_samples:
        back_part = unique_blanks[len(unique_blanks) - num_back_samples:]  # Extract the back part
        back_samples = random.sample(back_part, min(len(back_part), num_back_samples))  # Random sample
    else:
        back_samples = []
    
    # Combine the samples and ensure the total number does not exceed max_count
    sampled_blanks = front_samples + back_samples
    sampled_blanks = sampled_blanks[:max_count]
    
    return sampled_blanks

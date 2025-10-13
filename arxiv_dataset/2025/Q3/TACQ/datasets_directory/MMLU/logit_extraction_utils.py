import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

def aggregate_hidden_states_for_last_token(hidden_states_tuple, verbose=True):
    aggregated_hidden_state_for_one_generation = []
    for layer_state in hidden_states_tuple:
        aggregated_hidden_state_for_one_generation.append(layer_state[:,-1,:].clone())
    aggregated_hidden_state_for_one_generation = torch.stack(aggregated_hidden_state_for_one_generation, dim=0)
    return aggregated_hidden_state_for_one_generation

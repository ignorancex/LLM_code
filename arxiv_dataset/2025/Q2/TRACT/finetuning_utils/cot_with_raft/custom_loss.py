from transformers.loss.loss_utils import ForCausalLMLoss
import torch
from colorama import Fore, Style
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_label_from_sequence(labels):
    """
    For each sequence in labels (shape: [batch_size, seq_len]):
    - If the last element == -100, the label should be the second to last token before -100.
    - If the last token is not -100, it should be the second to last token
    This function returns a tensor of shape [batch_size] containing the extracted labels.
    This function will yield error if the sequence is cropped
    """
    batch_size = labels.shape[0]
    selected_labels = []
    pos = []

    for i in range(batch_size):
        seq = labels[i]
        # Get indices of all valid tokens
        valid_indices = (seq != -100).nonzero(as_tuple=True)[0]
        # Check the labels
        if len(valid_indices) < 2:
            raise ValueError(f"Not enough valid tokens in sequence {i} to select a second to last token.")
        # The second to last valid token index
        label_idx = valid_indices[-2]
        # Append the actual token at that index
        selected_labels.append(seq[label_idx])
        pos.append(label_idx)
    return torch.tensor(selected_labels), torch.tensor(pos)



def compute_raft_loss(outputs, labels, num_items_in_batch=None):
    """
    Custom loss function that adds an entropy regularization term to the base loss.
    This function is independent and does not reference the trainer instance.
    """
    num_seq = labels.size(0)

    # The following is customized for Mistral-7B-Instruct-v0.2
    score_to_indices = [28740, 28750, 28770, 28781, 28782]
    score_grids = [1.0, 2.0, 3.0, 4.0, 5.0]
    indices_to_scores = {
        28740: 1.0,
        28750: 2.0,
        28770: 3.0,
        28781: 4.0,
        28782: 5.0,
    }
    
    # The following is customized for LLama-3.1-8B-Instruct
    #  score_to_indices = [16, 17, 18, 19, 20]
    #  score_grids = [1.0, 2.0, 3.0, 4.0, 5.0]
    #  indices_to_scores = {
    #      16: 1.0,
    #      17: 2.0,
    #      18: 3.0,
    #      19: 4.0,
    #      20: 5.0,
    #  }

    logits = outputs.logits

    # Step 1: Collect the labels for the score
    # the size of labels is (batch_size, seq_len). calculate the effective input length by counting the number of non-padding tokens
    # Score position is the second to last non-padding token

    score_label_token_ids, score_pos = extract_label_from_sequence(labels)
    score_labels = [indices_to_scores[token_id.item()] for token_id in score_label_token_ids]

    score_labels = torch.tensor(score_labels, device=labels.device, dtype=logits.dtype)

    # Step 2: Mask out the score label from LM loss
    # logger.info(f"{Fore.GREEN}The sequence length is:{Style.RESET_ALL} {labels.size(1)}")
    # logger.info(f"The score position is: {score_pos}")
    labels[torch.arange(labels.size(0)), score_pos] = -100

    # Print the full labels without truncation

    # Step 3: Compute the LM loss
    # TODO: The num_items_in_batch is wrong since we mask out the score label. It should be substracted by something multiply by the world size?
    lm_loss = ForCausalLMLoss(
        logits = logits, 
        labels = labels,
        vocab_size = logits.size(-1),
        num_items_in_batch = num_items_in_batch - num_seq, # TODO: Modify this
    )

    # Step 4: Compute the score loss
    # Seq len 5
    # Token pos: 0 1 2 3 4
    # Input    : A B C D E
    # Is score : x x x v x
    # Predict  : B C D E - 
    # We take -1 due to the shift between input and output
    score_logits = logits[torch.arange(logits.size(0)), score_pos - 1, :]
    probs = torch.softmax(score_logits, dim=-1) # Shape: (batch_size, vocab_size)
    score_grid_probs = probs[..., score_to_indices].contiguous() # 
    # Compute the weighted sum of the score
    weighted_scores = torch.sum(
        score_grid_probs * torch.tensor(score_grids, device=probs.device, dtype = score_logits.dtype),
        dim = -1,
        keepdim = False,
    )

    # logger.info(f"{Fore.GREEN}score_label_token_ids:{Style.RESET_ALL} {score_label_token_ids}")
    # logger.info(f"{Fore.GREEN}score_labels:{Style.RESET_ALL} {score_labels}")
    # logger.info(f"{Fore.GREEN}score_grid_probs:{Style.RESET_ALL} {score_grid_probs}")
    # logger.info(f"{Fore.GREEN}weighted_scores:{Style.RESET_ALL} {weighted_scores}")

    # Compute the MSE loss
    score_loss = torch.nn.functional.mse_loss(
        input = weighted_scores, 
        target = score_labels,
        reduction = 'sum' if num_items_in_batch is None else 'mean',
    )

    if num_items_in_batch is not None:
        score_loss = score_loss / num_seq # TODO: This should be the number of sequences in the whole batch (I am not sure whether we should consider the world size)
    # TODO: Find a way to log the loss
    loss = lm_loss +  1.0 * score_loss
    print(f"LM loss: {Fore.BLUE}{lm_loss.item():.4f}{Style.RESET_ALL}, Score loss: {Fore.BLUE}{score_loss.item():.4f}{Style.RESET_ALL}")
    return loss


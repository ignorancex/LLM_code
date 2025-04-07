from transformers.loss.loss_utils import ForCausalLMLoss
import torch
from colorama import Fore, Style
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_raft_loss_no_lm(outputs, labels, num_items_in_batch=None):
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
    loss = score_loss
    print(f"Score loss: {Fore.BLUE}{score_loss.item():.4f}{Style.RESET_ALL}")
    return loss


def compute_raft_loss_with_pairwise(outputs, labels, num_items_in_batch=None):
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
    score_labels = []
    score_batch_idx = []
    new_score_pos = []
    for batch_idx, token_id in enumerate(score_label_token_ids):
        if token_id.item() in indices_to_scores:
            score_labels.append(indices_to_scores[token_id.item()])
            score_batch_idx.append(batch_idx)
            new_score_pos.append(score_pos[batch_idx])
        else:
            score_pos[batch_idx] = 0 # The first token is always masked out, so this will not affect the loss

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
    if len(score_batch_idx) != 0:
        new_score_pos = torch.tensor(new_score_pos, device=labels.device, dtype=torch.long)
        score_logits = logits[torch.tensor(score_batch_idx), new_score_pos - 1, :]
        probs = torch.softmax(score_logits, dim=-1) # Shape: (batch_size, vocab_size)
        score_grid_probs = probs[..., score_to_indices].contiguous() # 
        # Compute the weighted sum of the score
        weighted_scores = torch.sum(
            score_grid_probs * torch.tensor(score_grids, device=probs.device, dtype = score_logits.dtype),
            dim = -1,
            keepdim = False,
        )

        logger.info(f"{Fore.GREEN}score_label_token_ids:{Style.RESET_ALL} {score_label_token_ids}")
        logger.info(f"{Fore.GREEN}score_labels:{Style.RESET_ALL} {score_labels}")
        logger.info(f"{Fore.GREEN}score_grid_probs:{Style.RESET_ALL} {score_grid_probs}")
        logger.info(f"{Fore.GREEN}weighted_scores:{Style.RESET_ALL} {weighted_scores}")

        # Compute the MSE loss
        score_loss = torch.nn.functional.mse_loss(
            input = weighted_scores, 
            target = score_labels,
            reduction = 'sum' if num_items_in_batch is None else 'mean',
        )

        if num_items_in_batch is not None:
            score_loss = score_loss / len(score_batch_idx) # TODO: This should be the number of sequences in the whole batch (I am not sure whether we should consider the world size)
    else:
        score_loss = torch.tensor(0.0, device=labels.device, dtype=logits.dtype)
    # TODO: Find a way to log the loss
    loss = lm_loss +  1.0 * score_loss
    print(f"LM loss: {Fore.BLUE}{lm_loss.item():.4f}{Style.RESET_ALL}, Score loss: {Fore.BLUE}{score_loss.item():.4f}{Style.RESET_ALL}")
    return loss

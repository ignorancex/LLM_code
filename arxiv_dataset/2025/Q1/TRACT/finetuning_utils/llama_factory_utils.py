# This file should be placed in the same directory as the src/llamafactory/train/sft/workflow.py file 
import torch
from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask


# TODO: To incorporate this loss in the SFT trainer, we need to manually modify src/llamafactory/train/sft/trainer.py in LlamaFactory
# For example
###    trainer = CustomSeq2SeqTrainer(
###        model=model,
###        args=training_args,
###        finetuning_args=finetuning_args,
###        data_collator=data_collator,
###        callbacks=callbacks,
###        compute_loss_func = compute_entropy_loss, <-- Add this line
###        **dataset_module,
###        **tokenizer_module,
###        **metric_module,
###    )



def compute_raft_loss(outputs, labels, num_items_in_batch=None):
    """
    Custom loss function that adds an entropy regularization term to the base loss.
    This function is independent and does not reference the trainer instance.
    """
    # TODO: Check the num_items_in_batch parameter since this may be wrong

    logits = outputs.logits
    #  # Calcaulte the language model auto-regressive loss
    #  loss = ForCausalLMLoss(
    #      logits = logits, 
    #      labels = labels,
    #      vocab_size = logits.size(-1),
    #      num_items_in_batch = num_items_in_batch
    #  )

    # TODO: The score's indices should be double checked
    score_to_indices = [28740, 28750, 28770, 28781, 28782]
    score_grids = [1.0, 2.0, 3.0, 4.0, 5.0]
    indices_to_scores = {
        28740: 1.0,
        28750: 2.0,
        28770: 3.0,
        28781: 4.0,
        28782: 5.0,
    }

    # Select the indices. 
    # TODO: The indices for the seq dimension needs to be double-checked since this may be related with whether the EOS token loss should be considered 

    
    # print(f"Labels: {labels}")
    
    label_scores = torch.tensor(
        [indices_to_scores[label.item()] for label in labels], 
        device=labels.device,
        # dtype=logits.dtype,
    )

    logits = logits[..., -2, :] # Only get the logits for the -2nd token. The last token is the EOS token
    probs = torch.softmax(logits, dim=-1) # Shape: (batch_size, vocab_size)
    score_grid_probs = probs[..., score_to_indices].contiguous() # 

    # Compute the weighted sum of the score
    weighted_scores = torch.sum(
        score_grid_probs * torch.tensor(score_grids, device=probs.device),
        dim = -1,
        keepdim = True,
    )

    # Compute the MSE loss
    loss = torch.nn.functional.mse_loss(
        input = weighted_scores, 
        target = label_scores,
        reduction = 'sum' if num_items_in_batch is None else 'mean',
    )

    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch

    return loss


class SFTDataCollatorWith4DAttentionMaskForRegression(SFTDataCollatorWith4DAttentionMask):
    def __call__(self, features):
        features = super().__call__(features)

        def extract_label_from_sequence(labels):
            """
            For each sequence in labels (shape: [batch_size, seq_len]):
            - If the last element == -100, the label should be the second to last token before -100.
            - If the last token is not -100, it should be the second to last token.

            This function returns a tensor of shape [batch_size] containing the extracted labels.
            """
            batch_size = labels.shape[0]
            selected_labels = []

            for i in range(batch_size):
                seq = labels[i]
                # Get indices of all valid tokens
                valid_indices = (seq != -100).nonzero(as_tuple=True)[0]
                if len(valid_indices) < 2:
                    raise ValueError(f"Not enough valid tokens in sequence {i} to select a second to last token.")

                # The second to last valid token index
                label_idx = valid_indices[-2]

                # Append the actual token at that index
                selected_labels.append(seq[label_idx])

            return torch.stack(selected_labels)


        # Change the labels to the last token

        features['labels'] = extract_label_from_sequence(features['labels'])

        # The labels returned here are the 'token_idx' of the regression target, not the actual target.
        # TODO: Consider if we should return the actual target instead
        return features
"""This file calculates different types of loss
"""
import torch
import numpy as np

# Calculates loss (sum or average) for a batch
def multi_loss_fn(model, batch, task, reduc="mean"):
    # Set loss = CE ,reduction = sum or mean (default)
    if reduc == "mean":
        loss = torch.nn.CrossEntropyLoss()
    else:
        loss = torch.nn.CrossEntropyLoss(reduction='sum')
    # Pre-process batch and extract prediction
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targ = batch['labels']
    pred = model(input_ids, attention_mask=attention_mask, labels=targ)[1]
    # Reshape original predicted target
    if task == "wiki":
        # Remove last prediction in sequence (shift with target)
        pred = pred[:, :-1]
        targ = targ[:, 1:]  # Shift prediction 1 token to the right
    pred_reshape = pred[targ != -100]
    targ_reshape = targ[targ != -100]
    # Calculate loss
    loss_result = loss(pred_reshape, targ_reshape)
    return (loss_result)


# Calculates loss and accuracy for a batch
@torch.no_grad()
def loss_acc_batch_fn(model, tokenizer, batch, task):
    # --------- Loss -------------
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targ = batch['labels']
    # get prediction and reshape
    pred = model(input_ids, attention_mask=attention_mask, labels=targ)[1]
    if task == "wiki":
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift prediction 1 token to the right
    pred_reshape = pred[targ != -100]
    targ_reshape = targ[targ != -100]
    # Calculate loss
    loss_result = loss(pred_reshape, targ_reshape)
    # ---------Accuracy----------
    if task == "wiki":
        acc = 1
    else:
        # Calculate Accuracy
        generated_ids_pred = model.generate(batch["input_ids"], max_length=40)
        targ_raw = batch["decoder_raw"]
        pred_raw = tokenizer.batch_decode(
            generated_ids_pred, skip_special_tokens=True)
        pred_raw = [p.lower() for p in pred_raw]
        targ_raw = [p.lower() for p in targ_raw]
        acc = (np.array(pred_raw) == np.array(targ_raw)).sum()
    return {
        "acc_sum": acc,
        "nll_sum": loss_result,
        "n_tokens": pred_reshape.shape[0]
    }

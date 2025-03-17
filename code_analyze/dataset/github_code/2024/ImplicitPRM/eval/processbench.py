import os
import json
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from accelerate import Accelerator
from sklearn.metrics import roc_curve

###############################
# PART 1: SIGMOID & EVALUATION
###############################
def sigmoid(x):
    """Element-wise sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def find_threshold_sigmoid(data, num_thresholds=10000):
    """
    1. Ensure each record has d['reward'] as a list of raw scores.
    2. Transform each score via sigmoid -> d['sigmoid_scores'].
    3. Search thresholds in [0..1], pick best by a two-group F1-like measure:
       - label = -1 => "correct_data"
       - label != -1 => "error_data"
       * Predict the first step i where score_i < threshold (else -1).
       * Then measure accuracies in each group => compute 2*(a1*a2)/(a1+a2).
    Returns (best_threshold, best_f1).
    """
    # Sigmoid transform
    for d in data:
        d['sigmoid_scores'] = [sigmoid(s) for s in d['reward']]

    thresholds = np.linspace(0, 1, num_thresholds)

    best_threshold = None
    best_f1 = 0.0

    for t in thresholds:
        # Predict for each record
        for d in data:
            pred_step = -1
            for i, sc in enumerate(d['sigmoid_scores']):
                if sc < t:
                    pred_step = i
                    break
            d['match'] = (d['label'] == pred_step)

        correct_data = [x for x in data if x['label'] == -1]
        error_data   = [x for x in data if x['label'] != -1]

        # If one group is empty, skip
        if len(correct_data) == 0 or len(error_data) == 0:
            continue

        acc_1 = sum(x['match'] for x in correct_data) / len(correct_data)
        acc_2 = sum(x['match'] for x in error_data)   / len(error_data)

        if (acc_1 + acc_2) > 0:
            f1_metric = 2.0 * acc_1 * acc_2 / (acc_1 + acc_2)
        else:
            f1_metric = 0.0

        if f1_metric > best_f1:
            best_f1 = f1_metric
            best_threshold = t

    return best_threshold, best_f1

###############################
# PART 2: INFERENCE LOGIC
###############################
BATCH_SIZE = 2  # Default batch size; can be overridden via argparse
COEF = 0.001

def setup_accelerator_and_models(model_path, ref_model_path, tokenizer_path):
    """
    Initializes Accelerator (for multi-GPU/distributed)
    and loads main/ref models + tokenizer.
    Returns (accelerator, tokenizer, model, ref_model)
    """
    accelerator = Accelerator()  # auto-detects GPUs/CPUs

    # Load models & tokenizer
    print("Loading main model from:", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Loading reference model from:", ref_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path)

    print("Loading tokenizer from:", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Prepare for multi-GPU/CPU
    model, ref_model = accelerator.prepare(model, ref_model)
    return accelerator, tokenizer, model, ref_model

def get_logps(model, inputs):
    """
    Returns per-token log probabilities for a batch of inputs:
    inputs = {"input_ids", "attention_mask", "labels"}
    """
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift by 1 for standard causal LM
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["labels"][:, 1:].clone().long()

    # Replace -100 with 0 to avoid gather error
    shift_labels[shift_labels == -100] = 0

    # log-softmax
    log_probs = shift_logits.log_softmax(-1)  # (batch_size, seq_len-1, vocab_size)
    per_token_logps = torch.gather(
        log_probs, dim=2, index=shift_labels.unsqueeze(2)
    ).squeeze(2)  # (batch_size, seq_len-1)
    return per_token_logps

def build_item_tensors(item, tokenizer):
    """
    Prepares a single data item for the model:
      - item['query'], item['answer'] (list of steps), ...
    Returns (input_ids, attention_mask, labels, step_last_tokens).
    """
    # Build full conversation
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": item["query"]},
            {"role": "assistant", "content": "\n\n".join(item["answer"])}
        ],
        tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    # shape: (1, seq_len) => we want (seq_len,)
    input_ids = input_ids.squeeze(0)
    attention_mask = (input_ids != tokenizer.pad_token_id)

    # Step boundaries
    step_last_tokens = []
    for step_num in range(len(item["answer"]) + 1):
        conv = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["query"]},
                {"role": "assistant", "content": "\n\n".join(item["answer"][:step_num])}
            ],
            tokenize=False, add_generation_prompt=False
        ).strip()
        if step_num != 0 and step_num != len(item["answer"]):
            conv += "\n\n"
        conv_ids = tokenizer.encode(conv, add_special_tokens=False)
        step_last_tokens.append(len(conv_ids) - 2)

    labels = input_ids.clone()  # for standard LM
    return input_ids, attention_mask, labels, step_last_tokens

def collate_fn(batch_items):
    """
    Collate function for batching. Each element:
      (input_ids, attention_mask, labels, step_last_tokens, raw_item)
    We pad them to the same seq_len within the batch.
    """
    max_len = max(x[0].shape[0] for x in batch_items)

    input_ids_list, attn_mask_list, labels_list = [], [], []
    step_positions_list, raw_items_list = [], []
    for (inp, attn, lab, step_pos, raw_item) in batch_items:
        pad_len = max_len - inp.shape[0]

        # Pad input_ids
        padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
        # Pad attention
        padded_attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
        # Pad labels
        padded_lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])

        input_ids_list.append(padded_inp.unsqueeze(0))
        attn_mask_list.append(padded_attn.unsqueeze(0))
        labels_list.append(padded_lab.unsqueeze(0))
        step_positions_list.append(step_pos)
        raw_items_list.append(raw_item)

    input_ids_batch = torch.cat(input_ids_list, dim=0)     # (batch_size, max_len)
    attention_mask_batch = torch.cat(attn_mask_list, dim=0)# (batch_size, max_len)
    labels_batch = torch.cat(labels_list, dim=0)           # (batch_size, max_len)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
        "step_positions": step_positions_list,
        "raw_items": raw_items_list
    }

def inference(args):
    """
    1. Loads data from input json
    2. Compute 'reward' for each item
    3. Write out to new JSON (with "reward" field)
    """
    accelerator, tokenizer, model, ref_model = setup_accelerator_and_models(
        args.model_path, args.ref_model_path, args.tokenizer_path
    )

    # Load dataset
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Possibly modify data items if needed
    for d in data:
        d["query"] = d["problem"]  
        d["answer"] = [f"Step {i+1}: " + step for i, step in enumerate(d["steps"])]

    # Prepare data tuples
    data_tuples = []
    for d in data:
        inp, attn, lab, steps = build_item_tensors(d, tokenizer)
        data_tuples.append((inp, attn, lab, steps, d))

    results = []
    for i in tqdm(range(0, len(data_tuples), args.batch_size)):
        batch_slice = data_tuples[i : i + args.batch_size]
        batch = collate_fn(batch_slice)

        input_ids  = batch["input_ids"].to(accelerator.device)
        attn_mask  = batch["attention_mask"].to(accelerator.device)
        labels     = batch["labels"].to(accelerator.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels
        }

        with torch.no_grad():
            main_logps = get_logps(model, model_inputs)
            ref_logps  = get_logps(ref_model, model_inputs)

        # raw_reward = difference
        raw_reward = main_logps - ref_logps  # (batch, seq_len-1)

        # For each item in the batch
        for idx_in_batch, (inp, attn, lab, step_pos, raw_item) in enumerate(batch_slice):
            seq_len = input_ids.shape[1]
            first_boundary = step_pos[0]
            mask = torch.zeros(seq_len - 1, dtype=torch.float, device=accelerator.device)
            mask[first_boundary:] = 1.0

            item_raw_reward = raw_reward[idx_in_batch]
            weighted_reward = args.coef * item_raw_reward * mask
            csum = weighted_reward.cumsum(dim=-1)

            gather_indices = torch.tensor(step_pos[1:], device=accelerator.device)
            gather_indices = torch.clamp(gather_indices, 0, seq_len-2)

            final_values = csum.gather(dim=-1, index=gather_indices)
            final_values_list = final_values.cpu().tolist()

            # Store in the raw item
            raw_item["reward"] = final_values_list
            results.append(raw_item)

    # Save results
    print(f"Writing output to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

def evaluate(args):
    """
    Load the predicted file (with 'reward'), apply find_threshold_sigmoid,
    optionally apply that threshold to other tasks or just measure final F1 in the same file.
    """
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # find best threshold on the given data
    best_th, best_f1 = find_threshold_sigmoid(data, num_thresholds=args.num_thresholds)
    print(f"Best threshold found: {best_th:.4f}, F1={best_f1:.4f}")

###############################
# MAIN & ARGPARSE
###############################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated script for inference (reward computation) and threshold-based evaluation."
    )
    # Common
    parser.add_argument("--mode", type=str, required=True,
                        choices=["inference", "evaluate"],
                        help="Choose between computing rewards (inference) or evaluating threshold-based F1.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSON file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSON file.")
    # Inference
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to main model (for inference).")
    parser.add_argument("--ref_model_path", type=str, default="",
                        help="Path to reference model.")
    parser.add_argument("--tokenizer_path", type=str, default="",
                        help="Path to tokenizer.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for inference.")
    parser.add_argument("--coef", type=float, default=COEF,
                        help="Scaling coefficient (COEF) for raw_reward.")
    # Evaluation
    parser.add_argument("--num_thresholds", type=int, default=1000,
                        help="Number of thresholds to check in [0..1].")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "inference":
        # Validate that we have model paths
        if not (args.model_path and args.ref_model_path and args.tokenizer_path):
            raise ValueError("Please specify --model_path, --ref_model_path, --tokenizer_path for inference mode.")
        inference(args)

    elif args.mode == "evaluate":
        evaluate(args)

if __name__ == "__main__":
    main()
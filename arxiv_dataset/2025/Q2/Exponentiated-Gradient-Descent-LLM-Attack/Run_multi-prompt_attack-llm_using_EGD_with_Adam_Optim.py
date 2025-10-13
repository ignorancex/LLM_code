#!/usr/bin/env python
# coding: utf-8

# IMPORTS and GLOBALS
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook  # If you need both
import warnings
import csv
from behavior import Behavior
from helper import get_model_path, load_model_and_tokenizer, get_embedding_matrix, get_tokens, create_one_hot_and_embeddings, get_nonascii_toks
import argparse


def get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor):
    # Mask out all the non-ascii tokens and then return
    # Step 1: Create a tensor with all the non_ascii tokens
    top_token_ids_tensor_2d = non_ascii_toks_tensor.unsqueeze(0).repeat(20, 1)
    
    # Step 2: Create a mask with the same shape as one_hot_adv, initialized to zero
    mask = torch.ones_like(one_hot_adv, dtype=torch.float16)

    # Step 3: Use token_ids_init to set the corresponding indices in the mask to 1
    # We use gather and scatter operations for this
    mask.scatter_(1, top_token_ids_tensor_2d, 0.0)

    # Step 4: Apply the mask to one_hot_adv
    masked_one_hot_adv = one_hot_adv * mask
    
    return masked_one_hot_adv


def calc_loss(model, embeddings_user, embeddings_adv, embeddings_target, targets):
    full_embeddings = torch.hstack([embeddings_user, embeddings_adv, embeddings_target])
    logits = model(inputs_embeds=full_embeddings).logits
    loss_slice_start = len(embeddings_user[0]) + len(embeddings_adv[0])
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
    return loss, logits[:, loss_slice_start-1:, :]
    # return loss, logits.to(torch.float16)



class EGDwithAdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-4, grad_clip_val=1.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, grad_clip_val=grad_clip_val)
        super(EGDwithAdamOptimizer, self).__init__(params, defaults)

        # Initialize state variables for each parameter group
        for group in self.param_groups:
            for param in group['params']:
                self.state[param] = {
                    'm': torch.zeros_like(param),  # First moment vector (momentum)
                    'v': torch.zeros_like(param),  # Second moment vector (variance)
                    't': 0                         # Time step
                }

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            # grad_clip_val = group['grad_clip_val']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # Retrieve state variables
                m = state['m']
                v = state['v']
                t = state['t']

                # Increment time step
                t += 1
                state['t'] = t

                # # Clip gradients to avoid exploding updates; Redundant step
                # grad = torch.clamp(grad, min=-grad_clip_val, max=grad_clip_val)

                # Update biased first moment estimate (m) and second moment estimate (v)     
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2) 
                # v = beta2 * v + (1 - beta2) * (grad ** 2)
                state['m'] = m
                state['v'] = v

                # Bias correction
                m_hat = m / (1 - math.pow(beta1, t)) # m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - math.pow(beta2, t)) # v_hat = v / (1 - beta2 ** t)

                # Adam-like modified gradient
                modified_grad = m_hat / (torch.sqrt(v_hat) + eps)

                with torch.no_grad():
                    # Exponentiated Gradient Descent update with Adam-like modifications
                    param.mul_(torch.exp(-lr * modified_grad))

                    # Clamp the parameter values to avoid extreme values
                    param.clamp_(min=1e-12, max=1e12)

                    # Normalize each row to sum up to 1, considering possible tensor shapes
                    if param.dim() > 1:
                        row_sums = param.sum(dim=1, keepdim=True) + 1e-10  # Add epsilon to avoid division by zero
                        param.div_(row_sums)
                    else:
                        # Handle 1D tensors or other cases if applicable
                        param.div_(param.sum() + 1e-10)

        return loss


# Try Micro-Batch Update on the one_hot_adv to resolve CUDA OOM issue
def multi_prompt_adversarial_suffix_optimization_micro_batch(
    model,
    harmful_behaviors,
    tokenizer,
    device,
    embed_weights,
    directories,
    batch_size,
    micro_batch_size,
    num_steps,
    step_size,
    initial_coefficient=1e-5,
    final_coefficient=1e-3):

    user_prompt_tokens_list = [get_tokens(user_prompt, tokenizer=tokenizer, device=device) for user_prompt, _ in harmful_behaviors]
    target_tokens_list = [get_tokens(target, tokenizer=tokenizer, device=device)[1:] for _, target in harmful_behaviors]

    csv_filename = f'{directories[0]}/stats_multiprompt_using_EGD({len(harmful_behaviors)}_behaviors)({num_steps}_steps)({batch_size}_batchsize).csv'

    non_ascii_toks = get_nonascii_toks(tokenizer, device=device).tolist()
    non_ascii_toks_tensor = torch.tensor(non_ascii_toks).to(device=device)

    one_hot_adv = F.softmax(torch.rand(20, embed_weights.shape[0], dtype=torch.float16).to(device=device), dim=1).to(embed_weights.dtype)
    one_hot_adv.requires_grad_()
    # Initialize the optimizer and scheduler
    optimizer = EGDwithAdamOptimizer([one_hot_adv], lr=step_size)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    # Initialize the masked_one_hot_adv and others
    masked_one_hot_adv = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor=non_ascii_toks_tensor)
    max_values = torch.max(masked_one_hot_adv, dim=1)
    adv_token_ids = masked_one_hot_adv.argmax(dim=1)
    one_hot_discrete = F.one_hot(adv_token_ids, num_classes=embed_weights.shape[0]).to(embed_weights.dtype)

    column_names = ['epoch', 'learning_rate', 'entropy_term', 'kl_divergence_term', 'continuous_loss', 'discrete_loss']
    column_names.extend([f'max_{i}' for i in range(1, 21)])
    column_names.extend([f'token_id_{i}' for i in range(1, 21)])
    df = pd.DataFrame(columns=column_names)

    best_discrete_loss = np.inf
    best_discrete_loss_epoch = 0
    effective_adv_one_hot = None
    eps = 1e-12

    for epoch_no in tqdm(range(num_steps)):
        indices = np.random.permutation(len(harmful_behaviors))
        reg_coefficient = initial_coefficient * (final_coefficient / initial_coefficient) ** (epoch_no / (num_steps - 1))

        epoch_continuous_losses, epoch_discrete_losses, epoch_entropy_terms, epoch_kl_terms = [], [], [], []

        for batch_start in range(0, len(indices), batch_size):
            batch_ids = indices[batch_start:batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            log_one_hot = torch.log(one_hot_adv.to(dtype=torch.float32) + eps)
            entropy_term = (-one_hot_adv * (log_one_hot - 1)).sum() * reg_coefficient
            kl_divergence_term = -torch.log(torch.max(one_hot_adv, dim=1).values + eps).sum() * reg_coefficient
            entropy_term = entropy_term.detach()
            kl_divergence_term = kl_divergence_term.detach()

            epoch_entropy_terms.append(entropy_term.item())
            epoch_kl_terms.append(kl_divergence_term.item())

            with torch.no_grad():
                masked_one_hot_adv = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor=non_ascii_toks_tensor)
                max_values = torch.max(masked_one_hot_adv, dim=1)
                adv_token_ids = masked_one_hot_adv.argmax(dim=1)
                one_hot_discrete = F.one_hot(adv_token_ids, num_classes=embed_weights.shape[0]).to(embed_weights.dtype)

            batch_continuous_losses, batch_discrete_losses = [], []

            for micro_start in range(0, len(batch_ids), micro_batch_size):
                micro_ids = batch_ids[micro_start:micro_start + micro_batch_size]
                micro_loss = 0.0

                for idx in micro_ids:
                    user_tokens = user_prompt_tokens_list[idx]
                    target_tokens = target_tokens_list[idx]

                    _, emb_user = create_one_hot_and_embeddings(user_tokens, embed_weights, device=device)
                    _, emb_target = create_one_hot_and_embeddings(target_tokens, embed_weights, device=device)
                    one_hot_target, _ = create_one_hot_and_embeddings(target_tokens, embed_weights, device=device)

                    emb_adv = (one_hot_adv @ embed_weights).unsqueeze(0)
                    emb_adv_discrete = (one_hot_discrete @ embed_weights).unsqueeze(0)

                    ce_loss, _ = calc_loss(model, emb_user, emb_adv, emb_target, one_hot_target)
                    with torch.no_grad():
                        disc_loss, _ = calc_loss(model, emb_user, emb_adv_discrete, emb_target, one_hot_target)

                    regularized_loss = (ce_loss - entropy_term + kl_divergence_term) / (batch_size / micro_batch_size)
                    micro_loss += regularized_loss

                    batch_continuous_losses.append(ce_loss.item())
                    batch_discrete_losses.append(disc_loss.item())

                    del emb_user, emb_target, one_hot_target, emb_adv, emb_adv_discrete, ce_loss, disc_loss

                micro_loss.backward()   # type: ignore
                del micro_loss
                torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=1.0) # type: ignore
            optimizer.step()

            epoch_continuous_losses.append(np.mean(batch_continuous_losses))
            epoch_discrete_losses.append(np.mean(batch_discrete_losses))

            import gc
            gc.collect()
            torch.cuda.empty_cache()

        mean_continuous_loss = np.mean(epoch_continuous_losses)
        mean_discrete_loss = np.mean(epoch_discrete_losses)
        mean_entropy = np.mean(epoch_entropy_terms)
        mean_kl = np.mean(epoch_kl_terms)

        if mean_discrete_loss < best_discrete_loss:
            best_discrete_loss = mean_discrete_loss
            best_discrete_loss_epoch = epoch_no
            effective_adv_one_hot = one_hot_discrete.clone().detach()

        scheduler.step(mean_continuous_loss)

        max_values_array = max_values.values.detach().cpu().numpy()
        token_ids_array = adv_token_ids.detach().cpu().numpy()
        scheduler_lr = optimizer.param_groups[0]['lr']
        prepend_array = np.array([epoch_no, scheduler_lr, mean_entropy, mean_kl, mean_continuous_loss, mean_discrete_loss])
        row = np.concatenate((prepend_array, max_values_array, token_ids_array))
        new_row = pd.Series(row, index=df.columns)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        if (epoch_no + 1) % 10 == 0:
            df.to_csv(csv_filename, index=False)

    df.to_csv(csv_filename, index=False)
    return effective_adv_one_hot, best_discrete_loss, best_discrete_loss_epoch


# Try Mini-Batch Update on the one_hot_adv using EGD.

# def multi_prompt_adversarial_suffix_optimization(
#     model,
#     harmful_behaviors, 
#     tokenizer, 
#     device, 
#     embed_weights,
#     directories,
#     batch_size, 
#     num_steps, 
#     step_size, 
#     initial_coefficient=1e-5, 
#     final_coefficient=1e-3):
#     """
#     Perform multi-prompt adversarial suffix optimization.

#     Parameters:
#     - model: The model to use for optimization.
#     - dataset: The dataset containing harmful behaviors.
#     - harmful_behaviors: List of harmful behaviors to optimize over.
#     - tokenizer: Tokenizer used for tokenizing the user prompts and targets.
#     - device: The device (CPU/GPU) to run the computations on.
#     - batch_size: Number of samples in a batch.
#     - num_steps: Number of optimization steps (epochs).
#     - step_size: Learning rate for the optimizer.
#     - initial_coefficient: Initial regularization coefficient.
#     - final_coefficient: Final regularization coefficient.
#     """
    
#     # Prepare tokens from harmful behaviors
#     user_prompt_tokens_list = [get_tokens(user_prompt, tokenizer=tokenizer, device=device) for user_prompt, _ in harmful_behaviors]
#     target_tokens_list = [get_tokens(target, tokenizer=tokenizer, device=device)[1:] for _, target in harmful_behaviors]
    
#     #CSV filenames
#     csv_filename = f'{directories[0]}/stats_multiprompt_using_EGD({len(harmful_behaviors)}_behaviors)({num_steps}_steps)({batch_size}_batchsize).csv'
    
#     non_ascii_toks = get_nonascii_toks(tokenizer, device=device).tolist()
#     non_ascii_toks_tensor = torch.tensor(non_ascii_toks).to(device=device)   
    
#     # Initialize one-hot adversarial tokens and optimizer
#     one_hot_adv = F.softmax(torch.rand(20, embed_weights.shape[0], dtype=torch.float16).to(device=device), dim=1).to(embed_weights.dtype)
#     one_hot_adv.requires_grad_() 
#     # Initialize the masked_one_hot_adv and others
#     masked_one_hot_adv = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor=non_ascii_toks_tensor)
#     max_values = torch.max(masked_one_hot_adv, dim=1)
#     adv_token_ids = masked_one_hot_adv.argmax(dim=1)
#     one_hot_discrete = F.one_hot(adv_token_ids, num_classes=embed_weights.shape[0]).to(embed_weights.dtype)
#     # Initialize the optimizer
#     optimizer = EGDwithAdamOptimizer([one_hot_adv], lr=step_size)
#     # Initialize the learning_rate scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)

#     # DataFrame for logging optimization statistics
#     column_names = ['epoch', 'learning_rate', 'entropy_term', 'kl_divergence_term', 'continuous_loss','discrete_loss']
#     column_names.extend([f'max_{i}' for i in range(1, 21)])
#     column_names.extend([f'token_id_{i}' for i in range(1, 21)])
#     df = pd.DataFrame(columns=column_names)

#     # Best discrete loss initialization
#     best_discrete_loss = np.inf
#     best_discrete_loss_epoch = 0
#     effective_adv_one_hot = one_hot_discrete
#     # Used for regularization coefficient
#     eps = 1e-12
    
#     # Start optimization process
#     for epoch_no in tqdm(range(num_steps)):
#         indices = np.random.permutation(len(harmful_behaviors))

#         # Annealing regularization coefficient
#         reg_coefficient = initial_coefficient * (final_coefficient / initial_coefficient) ** (epoch_no / (num_steps - 1))
        
#         # Track per-epoch metrics
#         epoch_continuous_losses, epoch_discrete_losses = [], [] 
#         epoch_entropy_terms, epoch_kl_terms = [], []

#         for batch_start in range(0, len(indices), batch_size):
#             optimizer.zero_grad()
#             batch_ids = indices[batch_start:batch_start+batch_size]

#             # Compute entropy and KL divergence terms
#             log_one_hot = torch.log(one_hot_adv.to(dtype=torch.float32) + eps)
#             entropy_term = -one_hot_adv * (log_one_hot - 1)
#             entropy_term = entropy_term.sum() * reg_coefficient

#             kl_divergence_term = -torch.log(torch.max(one_hot_adv, dim=1).values + eps).sum()
#             kl_divergence_term *= reg_coefficient

#             epoch_entropy_terms.append(entropy_term.item())
#             epoch_kl_terms.append(kl_divergence_term.item())
            
#             masked_one_hot_adv = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor=non_ascii_toks_tensor)
#             max_values = torch.max(masked_one_hot_adv, dim=1)
#             adv_token_ids = masked_one_hot_adv.argmax(dim=1)
#             one_hot_discrete = F.one_hot(adv_token_ids, num_classes=embed_weights.shape[0]).to(embed_weights.dtype)

#             total_loss = 0.0
#             batch_continuous_losses, batch_discrete_losses = [], []
#             for idx in batch_ids:
#                 user_tokens = user_prompt_tokens_list[idx]
#                 target_tokens = target_tokens_list[idx]

#                 _, emb_user = create_one_hot_and_embeddings(user_tokens, embed_weights, device=device)
#                 _, emb_target = create_one_hot_and_embeddings(target_tokens, embed_weights, device=device)
#                 one_hot_target, _ = create_one_hot_and_embeddings(target_tokens, embed_weights, device=device)

#                 emb_adv = (one_hot_adv @ embed_weights).unsqueeze(0)
#                 emb_adv_discrete = (one_hot_discrete @ embed_weights).unsqueeze(0)

#                 ce_loss, _ = calc_loss(model, emb_user, emb_adv, emb_target, one_hot_target)
#                 with torch.no_grad():
#                     disc_loss, _ = calc_loss(model, emb_user, emb_adv_discrete, emb_target, one_hot_target)

#                 regularized_loss = ce_loss - entropy_term + kl_divergence_term
#                 total_loss += regularized_loss

#                 batch_continuous_losses.append(ce_loss.item())
#                 batch_discrete_losses.append(disc_loss.item())

#             epoch_continuous_losses.append(np.mean(batch_continuous_losses))
#             epoch_discrete_losses.append(np.mean(batch_discrete_losses))
#             # Compute gradients and update one_ho_adv
#             total_loss /= batch_size
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=1.0) # type: ignore
#             optimizer.step()
#             # Free any leftover memory from batch
#             del total_loss
#             torch.cuda.empty_cache()

#         mean_continuous_loss = np.mean(epoch_continuous_losses)
#         mean_discrete_loss = np.mean(epoch_discrete_losses)
#         mean_entropy = np.mean(epoch_entropy_terms)
#         mean_kl = np.mean(epoch_kl_terms)

#         # Save best discrete loss
#         if mean_discrete_loss < best_discrete_loss:
#             best_discrete_loss = mean_discrete_loss
#             best_discrete_loss_epoch = epoch_no
#             effective_adv_one_hot = one_hot_discrete

#         scheduler.step(mean_continuous_loss)

#         # Log results
#         max_values_array = max_values.values.detach().cpu().numpy()
#         token_ids_array = adv_token_ids.detach().cpu().numpy()
#         scheduler_lr = optimizer.param_groups[0]['lr']
#         prepend_array = np.array([epoch_no, scheduler_lr, mean_entropy, mean_kl, mean_continuous_loss, mean_discrete_loss])
#         row = np.concatenate((prepend_array, max_values_array, token_ids_array))
#         new_row = pd.Series(row, index=df.columns)
#         df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

#         # Save every 10 epochs
#         if (epoch_no + 1) % 10 == 0:
#             df.to_csv(csv_filename, index=False)

#     # Final save to CSV file
#     df.to_csv(csv_filename, index=False)
#     return effective_adv_one_hot, best_discrete_loss, best_discrete_loss_epoch



def generate_adversarial_outputs(
    model, 
    harmful_behaviors, 
    tokenizer, 
    embed_weights, 
    device, 
    effective_adv_one_hot, 
    num_tokens,
    num_steps, 
    batch_size, 
    directories, 
    best_discrete_loss, 
    best_discrete_loss_epoch):
    """
    Generate adversarial outputs and save them into JSON and JSONL files.

    Parameters:
    - model: The language model used for generation.
    - harmful_behaviors: List of tuples containing user prompts and target behaviors.
    - tokenizer: The tokenizer used for encoding and decoding.
    - embed_weights: The embedding weights of the model.
    - device: The device (CPU/GPU) where computations will be performed.
    - effective_adv_one_hot: The adversarial one-hot encoding to generate the adversarial suffix.
    - num_tokens: The number of tokens for the generated output.
    - batch_size: The batch size used for processing.
    - num_steps: Number of steps used in optimization.
    - json_filename: The filename where the results will be saved in JSON format.
    - directories: The directories where the output files will be saved.
    - best_discrete_loss: The best discrete loss encountered during optimization.
    - best_discrete_loss_epoch: The epoch at which the best discrete loss was found.
    """
    optimization_results = []

    # Create JSON and JSONL filename
    json_filename = f'{directories[1]}/outputs_multiprompt_using_EGD({len(harmful_behaviors)}_behaviors)({num_steps}_steps)({batch_size}_batchsize).json'
    output_filename = f'{directories[1]}/outputs_multiprompt_using_EGD({len(harmful_behaviors)}_behaviors)({num_steps}_steps)({batch_size}_batchsize).jsonl'
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `temperature` is set to.*")
    warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `top_p` is set to.*")

    # Loop through harmful behaviors
    for i, row in tqdm(enumerate(harmful_behaviors), total=len(harmful_behaviors), desc="Generating outputs"):
        user_prompt, target = row

        # Tokenize user prompt
        user_prompt_tokens = get_tokens(user_prompt, tokenizer=tokenizer, device=device)       
        # Get one-hot encodings and embedding vectors for the user prompts
        one_hot_inputs, _ = create_one_hot_and_embeddings(user_prompt_tokens, embed_weights, device=device)

        # Generate token ids
        inputs_token_ids = one_hot_inputs.argmax(dim=1)
        adv_token_ids = effective_adv_one_hot.argmax(dim=1)
        adv_suffix_list = str(adv_token_ids.cpu().numpy().tolist())
        adv_suffix_string = tokenizer.decode(adv_token_ids.cpu().numpy())
        final_string_ids = torch.hstack([inputs_token_ids, adv_token_ids])

        # Generate output using model
        generated_output = model.generate(final_string_ids.unsqueeze(0), 
                                          max_length=num_tokens, 
                                          pad_token_id=tokenizer.pad_token_id,
                                          do_sample=False)
        generated_output_string = tokenizer.decode(generated_output[0][:].cpu().numpy(), skip_special_tokens=True).strip()

        # Create the iteration result
        iteration_result = {
            "harmful-behaviour": user_prompt,
            "target": target,
            "suffix_token_ids": adv_suffix_list,
            "suffix_string": adv_suffix_string,
            "best_disc_loss": best_discrete_loss,
            "epch_at_best_disc_loss": best_discrete_loss_epoch,
            "outputs": generated_output_string
        }
        optimization_results.append(iteration_result)

        # Update the JSON File
        with open(json_filename, "w") as f:
            json.dump(optimization_results, f, indent=4, ensure_ascii=False)

        # Update the JSONL file for the results
        behavior = Behavior(user_prompt, adv_suffix_string, generated_output_string, "", "")
        with open(output_filename, 'a') as f:
            f.write(json.dumps(behavior.to_dict()) + '\n')

    print("Output generation and file updates completed.")


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Prompt Adversarial Attack on LLMs using EGD")
    parser.add_argument("--model_name", type=str, required=True, choices=["Llama2", "Falcon", "MPT", "Vicuna", "Mistral"])
    parser.add_argument("--dataset_name", type=str, required=True, choices=["AdvBench", "HarmBench", "JailbreakBench", "MaliciousInstruct"])
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=200)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tokens: int = 300 
    seed: int = 42
    step_size=0.1

    model_path = get_model_path(model_name=args.model_name)
    print(f"Using model: {args.model_name} from path: {model_path}")
    print(f"Using dataset: {args.dataset_name}")
    print(f"Batch size: {args.batch_size}, Num steps: {args.num_steps}")

    if seed is not None:
        torch.manual_seed(seed)

    model, tokenizer = load_model_and_tokenizer(
        model_path, low_cpu_mem_usage=True, use_cache=False, device=device # type: ignore
    )

    embed_weights = get_embedding_matrix(model)

    dataset_path: str = f"./data/{args.dataset_name}/harmful_behaviors.csv"
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        harmful_behaviors = list(reader)[:50]

    print(f"Loaded {len(harmful_behaviors)} harmful behaviors")

    directories = [
        f'./Multi-Prompt/CSV_Files/{args.model_name}/{args.dataset_name}',
        f'./Multi-Prompt/JSON_Files/{args.model_name}/{args.dataset_name}'
    ]
    create_directories(directories)
    micro_batch_size = args.batch_size//2 # Hopefully, it's 5
    effective_adv_one_hot, best_discrete_loss, best_discrete_loss_epoch = multi_prompt_adversarial_suffix_optimization_micro_batch(
        model=model, 
        harmful_behaviors=harmful_behaviors, 
        tokenizer=tokenizer, 
        device=device,    
        embed_weights=embed_weights,        
        directories=directories,
        batch_size=args.batch_size, 
        micro_batch_size = micro_batch_size,
        num_steps=args.num_steps,
        step_size=step_size
    )

    generate_adversarial_outputs(
        model=model, 
        harmful_behaviors=harmful_behaviors, 
        tokenizer=tokenizer, 
        embed_weights=embed_weights, 
        device=device,
        effective_adv_one_hot=effective_adv_one_hot, 
        num_tokens=num_tokens, 
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        directories=directories, 
        best_discrete_loss=best_discrete_loss, 
        best_discrete_loss_epoch=best_discrete_loss_epoch
    )


if __name__ == "__main__":
    main()

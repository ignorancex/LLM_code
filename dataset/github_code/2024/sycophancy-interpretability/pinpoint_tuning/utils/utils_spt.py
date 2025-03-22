#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-18 16:22
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-18 22:08
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import numpy as np
import os
from typing import List, Dict

import torch
import torch.nn as nn

from transformers import PreTrainedModel

from utils import logging_rank

"""
This is the code for supervised pinpoint tuning some heads in the LLM.

The three functions below are only different in the function names since the architecture of these models are the some, 
we seperate them in case there are some changes in the future or you want to add new models.
"""


def load_path_patching_results(path_patching_path: str, num_key_value_groups: int, train_topk: int = 32) -> Dict[int, List[int]]:
    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"

    path_patching_results = torch.load(os.path.join(path_patching_path, 'results.pt'), weights_only=True).numpy()
    logging_rank(f"Load path patching results from {path_patching_path}")

    all_heads = {}
    _, n_heads = path_patching_results.shape
    path_patching_results = path_patching_results.reshape(-1)

    # Heads with lower path patching results are selected, since they have 
    #   stronger negative influence on the model after being patched.
    min_idxs= np.argsort(path_patching_results)

    for _, idx in enumerate(min_idxs[:train_topk]):
        layer_id = idx // n_heads
        head_id = idx % n_heads
        cur_set = all_heads.get(layer_id, set())
        cur_set.add(head_id)
        all_heads[layer_id] = cur_set

    # Compute groups_to_update for key_value_heads
    all_groups = {}
    for layer, heads in all_heads.items():
        groups = set()
        for head in heads:
            if head is None:
                continue
            group = head // num_key_value_groups
            groups.add(group)
        all_groups[layer] = list(groups)

    return all_heads, all_groups


def freeze_modules(model: PreTrainedModel, path_patching_path: str, precise_level: int = 0, train_topk: int = 32, train_kv: bool = False) -> None:
    """Freeze some unimportant modules in Llama.

    Parameters
    ----------
    model : PreTrainedModel
        LLM model.
    path_patching_path : str
        Path to path patching results.
    precise_level : int, optional
        Precise level used in pinpoint tuning, higher means freeze more parameters, by default 0
    train_topk : int, optional
        How many heads to train, by default 32
    train_kv : bool, optional
        Whether to train key, value of heads in attentino, by default False
    """
    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"

    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    if hasattr(model.config, 'head_dim'):
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_attention_heads
    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    # Load path patching results
    all_heads, all_groups = load_path_patching_results(path_patching_path, num_key_value_groups, train_topk)

    def freeze_q_proj_hook(layer_module: nn.Module, heads_to_update: List[int]) -> None:
        """Freeze attention layer"""

        def hook(grad):
            # We initialize a new tensor to store the gradients.
            new_grad = torch.zeros_like(grad, device=grad.device)

            for head in heads_to_update:
                # We set the gradients of the current head to the corresponding
                #   gradients of the original model.
                new_grad[head * head_dim: (head + 1) * head_dim] = grad[head * head_dim: (head + 1) * head_dim]

            return new_grad
        
        layer_module.self_attn.q_proj.weight.register_hook(hook)
        return
    
    def freeze_kv_proj_hook(layer_module: nn.Module, groups_to_update: List[int]) -> None:
        """Freeze attention layer"""

        def hook(grad):
            # We initialize a new tensor to store the gradients.
            new_grad = torch.zeros_like(grad, device=grad.device)

            for head in groups_to_update:
                # We set the gradients of the current head to the corresponding
                #   gradients of the original model.
                new_grad[head * head_dim: (head + 1) * head_dim] = grad[head * head_dim: (head + 1) * head_dim]

            return new_grad

        layer_module.self_attn.k_proj.weight.register_hook(hook)
        layer_module.self_attn.v_proj.weight.register_hook(hook)
        return

    def freeze_o_proj_hook(layer_module: nn.Module, heads_to_update: List[int]) -> None:
        """Freeze attention layer"""

        def hook(grad):
            # We initialize a new tensor to store the gradients.
            new_grad = torch.zeros_like(grad, device=grad.device)

            for head in heads_to_update:
                # We set the gradients of the current head to the corresponding
                #   gradients of the original model.
                new_grad[:, head * head_dim: (head + 1) * head_dim] = grad[:, head * head_dim: (head + 1) * head_dim]

            return new_grad
        
        layer_module.self_attn.o_proj.weight.register_hook(hook)
        return
    
    # Keep the gradient of embedding layer outputs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    
    logging_rank("@@@ Start freezing modules...")

    # First freeze all parameters, then unfreeze the heads we want to train.
    for param in model.parameters():
        param.requires_grad = False

    assert precise_level <= 4, "precise_level should be less than 4"

    if precise_level <= 1:
        # Unfreeze embedding layer.
        model.lm_head.weight.requires_grad = True
        model.model.embed_tokens.weight.requires_grad = True

    # Keep the gradient of embedding layer and freeze using hook to avoid gradient BUG.
    # model.model.embed_tokens.weight.requires_grad = True

    # if precise_level > 1:
        # Freeze embedding layer.
        # def wte_hook(grad):
        #     return torch.zeros_like(grad, device=grad.device)
        
        # model.model.embed_tokens.weight.register_hook(wte_hook)

    for index, layer in enumerate(model.model.layers):
        
        # If the layer is not in all_heads, skip it.
        if index not in all_heads:
            continue

        update_mlp = True
        logging_rank(f">>> Masking for Layer: {index}. Heads to update: {all_heads[index]}")

        if precise_level <= 2 and update_mlp:
            
            # Unfreeze mlp layer.
            logging_rank(f"--- Unfreezing mlp for Layer: {index}...")
            for name, parameter in layer.named_parameters():
                if 'mlp' in name:
                    parameter.requires_grad = True

        if len(all_heads[index]) == 0:
            continue

        if precise_level <= 3:
            
            # Unfreeze self_attn.o_proj.
            logging_rank(f"--- Unfreezing self_attn.o_proj for Layer: {index}...")

            for name, parameter in layer.named_parameters():
                if 'self_attn.o_proj' in name:
                    parameter.requires_grad = True

            # Add hook to freeze self_attn.q_proj.
            freeze_o_proj_hook(layer, all_heads[index])

        if precise_level <= 4:

            # Unfreeze self_attn.q_proj.
            logging_rank(f"--- Unfreezing self_attn.q_proj for Layer: {index}...")
            for name, parameter in layer.named_parameters():
                if 'self_attn.q' in name:
                    parameter.requires_grad = True

            freeze_q_proj_hook(layer, all_heads[index])

            if train_kv:
                logging_rank(f">>> Masking for Layer: {index}. Groups to update: {all_groups[index]}")

                # Unfreeze self_attn.k_proj / v_proj.
                logging_rank(f"--- Unfreezing self_attn.k_proj / v_proj for Layer: {index}...")
                for name, parameter in layer.named_parameters():
                    if 'self_attn.k' in name or 'self_attn.v' in name:
                        parameter.requires_grad = True

                freeze_kv_proj_hook(layer, all_groups[index])

    return


def freeze_lora_modules(model: PreTrainedModel, path_patching_path: str, precise_level: int, train_topk: int, train_kv: bool) -> None:
    """Combine SPT and LoRA freezing"""

    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"

    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    if hasattr(model.config, 'head_dim'):
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_attention_heads
    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    # Load path patching results
    all_heads, all_groups = load_path_patching_results(path_patching_path, num_key_value_groups, train_topk)

    assert precise_level >= 3, "Currently only support precise_level=3 when using LoRA."

    def freeze_lora_A_hook(parameter: nn.Parameter, heads_to_update: List[int]):
        """Freeze the parameter for lora_A, used if freeze along input dimension"""

        def hook(grad):
            # We initialize a new tensor to store the gradients.
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                # We set the gradients of the current head to the corresponding
                #   gradients of the original model.
                new_grad[:, head * head_dim: (head + 1) * head_dim] = grad[:, head * head_dim: (head + 1) * head_dim]

            return new_grad
        
        parameter.register_hook(hook)

    def freeze_lora_B_hook(parameter: nn.Parameter, heads_to_update: List[int]):
        """Freeze the parameter for lora_B, used if freeze along output dimension"""

        def hook(grad):
            # We initialize a new tensor to store the gradients.
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                # We set the gradients of the current head to the corresponding
                #   gradients of the original model.
                new_grad[head * head_dim: (head + 1) * head_dim, :] = grad[head * head_dim: (head + 1) * head_dim, :]

            return new_grad
        
        parameter.register_hook(hook)

    def get_update_heads(parameter_name: str, all_update_heads: Dict[int, List[int]]):
        """Get the heads to update for the current layer"""

        for index in all_update_heads:
            if f"layers.{index}." in parameter_name:
                return all_update_heads[index]
        
        return None
    
    logging_rank("@@@ Start freezing parameters...")
    
    for name, parameter in model.named_parameters():
        """Loop over all parameters"""

        # Skip non-trainable parameters in LoRA + SPT setting
        if not parameter.requires_grad:
            logging_rank(f">>> Skipping {name}...")
            continue

        # If the layer is not in all_heads, skip it.
        if get_update_heads(name, all_heads) is None:
            logging_rank(f"--- Disabling gradients for {name}...")
            parameter.requires_grad = False
            continue

        # Enable o_proj if precise_level <= 3
        if precise_level <= 3:
            if 'lora_A' in name and 'o_proj' in name:
                logging_rank(f"||| Masking for Layer: {name}...")
                freeze_lora_A_hook(parameter, get_update_heads(name, all_heads))

        # Enable q_proj / k_proj / v_proj if precise_level <= 4
        if precise_level <= 4:
            if 'lora_B' in name and 'q_proj' in name:
                logging_rank(f"||| Masking for Layer: {name}...")
                freeze_lora_B_hook(parameter, get_update_heads(name, all_heads))

            if train_kv:
                if 'lora_B' in name and ('k_proj' in name or 'v_proj' in name):
                    logging_rank(f"||| Masking for Layer: {name}...")
                    freeze_lora_B_hook(parameter, get_update_heads(name, all_heads))

    return
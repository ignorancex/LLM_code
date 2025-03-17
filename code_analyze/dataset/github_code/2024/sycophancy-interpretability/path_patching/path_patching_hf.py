#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 14:21
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 17:00
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import argparse
import json
import os
import shutil

import torch
import plotly
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import PathPatchingDataset
from hook_functions import add_pre_module_hook, add_pre_module_hook_single_head
from utils import compute_metric, create_batch, show_path_patching_results


@torch.no_grad()
def path_patching_batch(model, tokenizer, batch_data, module_input_name, module_output_name, num_layers, num_attention_heads, head_dim):
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    # Create path patching data
    xr_toks, xr_mask = create_batch(batch_data, split="xr_toks", pad_token_id=tokenizer.pad_token_id)
    xc_toks, xc_mask = create_batch(batch_data, split="xc_toks", pad_token_id=tokenizer.pad_token_id)
    
    xr_toks = xr_toks.to(model.device)
    xc_toks = xc_toks.to(model.device)
    xr_mask = xr_mask.to(model.device)
    xc_mask = xc_mask.to(model.device)
    
    default_logit_diff = compute_metric(
        model,
        xr_toks,
        xr_mask,
        [item["predict_token_id"] for item in batch_data],
        [item["record_token_ids"] for item in batch_data],
    )

    print(default_logit_diff)

    Hr = {} # Hr stores the hidden states of Xr
    Hc = {} # Hc stores the hidden states of Xc

    # Please refer to 'Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small'
    # for the details of forward A / B / C.
    
    # Forward A: Record the activation of Xr
    hooks = []
    for i in range(num_layers):
        name = module_input_name.format(i=i)
        module = model.get_submodule(name)  # Get the module
        
        # Add hook to record the hidden states of Xr
        hook = add_pre_module_hook(
            module,
            name,
            Hr,
            -1,
            True,
        )
        hooks.append(hook)

    # Run forward pass to record
    _ = model(xr_toks.to(model.device), attention_mask=xr_mask.to(model.device))
    for hook in hooks:
        hook.remove()   # Remove the hook

    # Forward B: Record the activation of Xc
    hooks = []
    for i in range(num_layers):
        name = module_output_name.format(i=i)
        module = model.get_submodule(name)

        hook = add_pre_module_hook_single_head(
            module,
            name,
            Hc,
            -1,
            head_dim,
            True,
        )
        hooks.append(hook)

    logits = model(xc_toks.to(model.device), attention_mask=xc_mask.to(model.device)).logits
    print("Xc logits:")
    print(torch.topk(logits[0, -1], k=10))
    for hook in hooks:
        hook.remove()

    # Loop through all layers and heads
    for source_layer in tqdm(range(num_layers)):
        for source_head_idx in [None] + list(range(num_attention_heads)):
            if source_head_idx is None:
                continue

            # Forward C: In the forward pass of Xr, replace the hidden states of Xr with the hidden states of Xc
            #   to observe the effect of each attention head on the logits.
            hooks = []
            for j in range(source_layer+1, num_layers):
                name = module_input_name.format(i=j)
                module = model.get_submodule(name)
                hook = add_pre_module_hook(
                    module,
                    name,
                    Hr,
                    -1,
                    False,
                )
                hooks.append(hook)

            name = module_output_name.format(i=source_layer)
            module = model.get_submodule(name)
            hook = add_pre_module_hook_single_head(
                module,
                name,
                Hc,
                -1,
                head_dim,
                False,
                source_head_idx
            )
            hooks.append(hook)

            cur_logit_diff = compute_metric(
                model,
                xr_toks,
                xr_mask,
                [item["predict_token_id"] for item in batch_data],
                [item["record_token_ids"] for item in batch_data],
            )
            for hook in hooks:
                hook.remove()

            results[source_layer][source_head_idx] += (
                (cur_logit_diff - default_logit_diff) / default_logit_diff
            ).mean(dim=0)

    results *= 100

    return results * len(batch_data)


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser("Path Patching Arguments")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="./data/path_patching_data.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to use for path patching")

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size

    if model_path[-1] == '/':
        model_path = model_path[:-1]


    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get the name of module used in path patching
    path_patchin_config = json.load(open(os.path.join('configs', f"{model.config.model_type}.json"), 'r', encoding='utf-8'))
    module_input_name = path_patchin_config["module_input_name"]
    module_output_name = path_patchin_config["module_output_name"]

    # Construct path patching dataset
    full_data = PathPatchingDataset(
        data_path,
        tokenizer,
    )

    sample_num = args.sample_num if args.sample_num is not None else len(full_data)

    # Print some examples
    print(f"Xr from dataset:\n" + "-" * 30 + f"\n{full_data.xr[0]}")
    print()
    print(f"Xc from dataset:\n" + "-" * 30 + f"\n{full_data.xc[0]}")


    # Get the number of layers, attention heads and head dimension from config
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    if hasattr(model.config, "head_dim"):
        # In some cases, head_dim is not hidden_size // num_attention_heads, like Gemma.
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Create path_patching_dir to save results
    path_patching_dir = f'results/{os.path.basename(model_path)}'
    if not os.path.exists(path_patching_dir):
        os.makedirs(path_patching_dir)
        
    # Copy data_path to path_patching_dir
    shutil.copy(data_path, path_patching_dir)

    # Initialize results
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    cnt = 0
    for i in tqdm(range(0, sample_num, batch_size)):
        batch_data = [full_data[i] for i in range(i, min(i+batch_size, len(full_data)))]
        results += path_patching_batch(
            model,
            tokenizer,
            batch_data,
            module_input_name,
            module_output_name,
            num_layers,
            num_attention_heads,
            head_dim,
        )
        cnt += len(batch_data)

    results /= cnt
    indices = torch.topk(results.flatten(), k=16, largest=False).indices.detach().cpu().numpy()
    all_results = []
    for i in indices:
        all_results.append((i // num_attention_heads, i % num_attention_heads))
    print(f"Top 16 Attention Heads:")
    print(all_results)
    
    # save results
    torch.save(results.detach().cpu(), f"{path_patching_dir}/results.pt")

    # show attention head results
    fig = show_path_patching_results(
        results.detach().cpu().numpy(),
        title=f"Effect of patching (Heads->Final Residual Stream State) path",
        return_fig=True,
        show_fig=False,
        bartitle="% change in logit difference",
    )

    # Save the plot
    plotly.offline.plot(fig, filename=f'{path_patching_dir}/head_map.html')

    # Get attention scores of an example
    dialogue = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the highest mountain in the world?"},
        {"role": "assistant", "content": "Mount Everest."},
        {"role": "user", "content": "I don't think that's right, are you sure?"},
        {"role": "assistant", "content": "Apologies"}
    ]
    text = tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=False)

    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model(**inputs, output_attentions=True)
    attention_scores = outputs.attentions

    tokens = [tokenizer.decode(token).replace("▁", "_").replace("\n", "↵") for token in inputs.input_ids[0].detach().cpu().tolist()]
    font_prop = matplotlib.font_manager.FontProperties(size=10) 

    if not os.path.exists(f'{path_patching_dir}/attn_maps'):
        os.makedirs(f'{path_patching_dir}/attn_maps')

    # For Top-16 attention heads, save attention maps of the example
    for i, (layer_idx, head_idx) in tqdm(enumerate(all_results), total=16, desc="Saving attention maps"):
        plt.figure(figsize=(16, 16))
        
        plt.imshow(attention_scores[layer_idx][0, head_idx].detach().cpu().numpy(), cmap='Blues')
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontproperties=font_prop)
        plt.yticks(range(len(tokens)), tokens, fontproperties=font_prop)
        plt.colorbar(shrink=0.8)
        
        plt.savefig(f'{path_patching_dir}/attn_maps/{i:02d}_l{layer_idx}_h{head_idx}.png', bbox_inches='tight', dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
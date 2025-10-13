import os
import random

import einops
import numpy as np
import pickle as pkl
import torch
from tqdm import tqdm

from utils.prompt_utils import PromptGenerator
from utils.args_utils import get_prompt_template_args



def get_dataset_with_target_modality(args, target_modality):
    from data.multimodal_torchvision import MultimodalDataset

    assert target_modality in ["image", "text"], "Target Modality should be ['image', 'text']"

    args.modality_to_report = target_modality # ["image", "text"]

    args = get_prompt_template_args(args)
    
    random.seed(0)
    np.random.seed(0)
    prompt_generator = PromptGenerator(args)
    dataset = MultimodalDataset(args, "test", prompt_generator)

    return dataset


def get_config(args, model):
    if args.model_family in ["llava", "llava-onevision", "instructblip"]:
        N_LAYERS = len(model.language_model.model.layers)
        N_HEADS = model.config.text_config.num_attention_heads
        D_MODEL = model.config.text_config.hidden_size
        D_HEADS = D_MODEL // N_HEADS
    
    elif args.model_family in ["qwen"]:
        N_LAYERS = model.config.num_hidden_layers
        N_HEADS = model.config.num_attention_heads
        D_MODEL = model.config.hidden_size
        D_HEADS = D_MODEL // N_HEADS
    
    else:
        raise NotImplementedError(f"{args.model_str} is not supported yet.")
    
    return N_LAYERS, N_HEADS, D_MODEL, D_HEADS


def run_head_attribution(args, processor, model, dataset, target_modality, n_samples=20):
    N_LAYERS, N_HEADS, D_MODEL, D_HEADS = get_config(args, model)

    # Check if logit diff have been precomputed already
    output_dir = os.path.join(args.output_dir_prefix, f"{target_modality}_n{args.num_samples}")
    output_path = os.path.join(output_dir, "precompute_logit_diff.pkl")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        print("Found existing precompute_logit_diff.pkl")
        with open(output_path, "rb") as f:
            all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list = pkl.load(f)
        return all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list

    all_logit_diffs = np.zeros((n_samples, N_LAYERS, N_HEADS))

    all_total_logit_diffs = np.zeros((n_samples,))

    all_predictions_clean = []

    curr_num_samples = 0
    num_invalid = 0
    sample_idx = 0
    sample_idx_list = []

    pbar = tqdm(total=n_samples)

    while curr_num_samples < n_samples:

        prompt = dataset[sample_idx]['source']
        image = dataset[sample_idx]['image']

        if args.model_family in ["llava", "instructblip"]:
            caption_label_word = dataset[sample_idx]['choices'][dataset[sample_idx]['relative_caption_label']].title()
            image_label_word   = dataset[sample_idx]['choices'][dataset[sample_idx]['relative_image_label']].title()
        elif args.model_family in ["qwen", "llava-onevision"]:
            caption_label_word = dataset[sample_idx]['choices'][dataset[sample_idx]['relative_caption_label']].lower()
            image_label_word   = dataset[sample_idx]['choices'][dataset[sample_idx]['relative_image_label']].lower()

        caption_first_token_id = processor.tokenizer(caption_label_word, add_special_tokens=False)['input_ids'][0]
        image_first_token_id   = processor.tokenizer(image_label_word, add_special_tokens=False)['input_ids'][0]

        # Resample if the first tokens of image and caption are the same
        if caption_first_token_id == image_first_token_id:
            num_invalid += 1        
            sample_idx += 1
            continue
        
        clean_inputs = processor(text=prompt, images=image, return_tensors="pt")

        clean_inputs = {k:v.to(model.device) for k,v in clean_inputs.items()}

        # clean_inputs['pixel_values'] = clean_inputs['pixel_values'].squeeze(0)

        with model.trace(**clean_inputs) as tracer:
            with torch.no_grad():

                clean_cache_zs = {}
                for layer_idx in range(N_LAYERS):
                    # attention output for llama models needs to be reshaped to look at individual heads
                    if args.model_family in ["llava", "llava-onevision", "instructblip"]:
                        z = model.language_model.model.layers[layer_idx].self_attn.o_proj.input # dimensions [batch x seq x D_MODEL]
                    elif args.model_family in ["qwen"]:
                        z = model.model.layers[layer_idx].self_attn.o_proj.input # dimensions [batch x seq x D_MODEL]
                    
                    z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS)
                    for head_idx in range(N_HEADS):
                        clean_cache_zs[layer_idx, head_idx] = z_reshaped[:,:,head_idx,:].cpu().detach().save()
                
                if args.model_family in ["llava", "llava-onevision", "instructblip"]:
                    clean_logits = model.language_model.lm_head.output
                elif args.model_family in ["qwen"]:
                    clean_logits = model.lm_head.output

                clean_logit_diff = (
                    clean_logits[0, -1, image_first_token_id] - clean_logits[0, -1, caption_first_token_id]
                ).cpu().save()
                pred = clean_logits[:,-1,:].argmax(-1).detach().cpu().numpy().save()
        
        all_predictions_clean.append(pred)
        all_total_logit_diffs[curr_num_samples] = clean_logit_diff


        if args.model_family in ["llava", "llava-onevision", "instructblip"]:
            caption_candidate_embed = model.language_model.lm_head.state_dict()['weight'][caption_first_token_id].cpu()
            image_candidate_embed   = model.language_model.lm_head.state_dict()['weight'][image_first_token_id].cpu()
        elif args.model_family in ["qwen"]:
            caption_candidate_embed = model.lm_head.state_dict()['weight'][caption_first_token_id].cpu()
            image_candidate_embed   = model.lm_head.state_dict()['weight'][image_first_token_id].cpu()


        for layerIDX in range(N_LAYERS):

            if args.model_family in ["llava", "llava-onevision", "instructblip"]:
                o_proj = model.language_model.model.layers[layerIDX].self_attn.o_proj
            elif args.model_family in ["qwen"]:
                o_proj = model.model.layers[layerIDX].self_attn.o_proj

            for headIDX in range(N_HEADS):

                multi_hot_z = torch.zeros((D_MODEL,), dtype=torch.float16)
                multi_hot_z[headIDX * D_HEADS: (headIDX+1) * D_HEADS] = clean_cache_zs[(layerIDX, headIDX)][0,-1,:]
                multi_hot_z = multi_hot_z.to(model.device)

                if args.model_family in ["llava", "llava-onevision", "instructblip"]:
                    projected_outputs = o_proj(multi_hot_z).cpu()
                    logit_diff = torch.dot(image_candidate_embed, projected_outputs) - torch.dot(caption_candidate_embed, projected_outputs)
                elif args.model_family in ["qwen"]:
                    multi_hot_z = multi_hot_z.bfloat16()
                    projected_outputs = o_proj(multi_hot_z).cpu()
                    logit_diff = torch.dot(image_candidate_embed, projected_outputs) - torch.dot(caption_candidate_embed, projected_outputs)

                all_logit_diffs[curr_num_samples, layerIDX, headIDX] = logit_diff
        
        sample_idx_list.append(sample_idx)
        curr_num_samples += 1
        sample_idx += 1
        pbar.update(1)

    all_predictions_clean = np.concatenate(all_predictions_clean)

    with open(output_path, "wb") as f:
        pkl.dump(
            (
                all_logit_diffs, 
                all_total_logit_diffs, 
                all_predictions_clean, 
                sample_idx_list
            ), f
        )

    print(f"num_examples_with_same_caption_image_first_token: {num_invalid}")
    return all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list

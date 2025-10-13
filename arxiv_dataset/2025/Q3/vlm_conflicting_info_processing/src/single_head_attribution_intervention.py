import argparse
import os

import torch
from tqdm import tqdm
import pickle as pkl
import json
from nnsight import NNsight
import numpy as np
import matplotlib.pyplot as plt

from utils.args_utils import get_model_family, dict_to_object
from utils.model_utils import load_model_and_preprocess


from head_attribution.utils import (
    get_dataset_with_target_modality,
    run_head_attribution,
)
from head_attribution.intervention_utils import (
    get_sampled_dataset,
    get_accuracy,
)
from head_attribution.visualization_utils import (
    get_intervened_plot,
    get_unimodal_intervened_plot,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments():

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    args = {}
    
    # If --config is provided, load arguments from JSON
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            args.update(json.load(f))
    # Otherwise, use normal argparse
    else:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--work_dir", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--version", type=int, required=True)
        parser.add_argument("--precompute_version", type=int, required=True)
        parser.add_argument("--num_samples", type=int, required=True)

        parser.add_argument("--layer_id", type=int, required=True)
        parser.add_argument("--head_id", type=int, required=True)
        parser.add_argument("--alpha_lower_bound", type=float, required=True)
        parser.add_argument("--alpha_upper_bound", type=float, required=True)
        parser.add_argument("--alpha_interval", type=float, required=True)

        # Prompt-related arguments
        parser.add_argument("--caption_type", type=str, choices=["consistent", "inconsistent", "no_caption", "irrelevant", "text_only"], default="inconsistent")
        parser.add_argument("--order", type=str, choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"], default="icq")
        parser.add_argument(
            "--n_candidates", type=int, default=5,
            help="Number of answer candidates in the prompt."
        )
        parser.add_argument(
            "--is_explicit_helper", type=int, default=0,
            help="Whehter to have explicit helper 'Ignoring the image or caption, what is the label in xxx' in the prompt."
        )
        parser.add_argument(
            "--is_assistant_prompt", type=int, default=1,
            help="'USER' and 'ASSISTANT' which specifices user an model in the prompt."
        )
        parser.add_argument("--use_pointers", type=int, default=1, help="Whether to use 'Caption: '/ 'Image: ' for caption and image.")
        parser.add_argument("--batch_size", type=int, default=1)

        args.update(vars(parser.parse_args()))

    args = dict_to_object(args)
    args.model_family = get_model_family(args)

    return args


def get_args():
    args = parseArguments()

    args.model_str = args.model_name
    
    args.model_family = get_model_family(args)
    
    args.output_dir_prefix = os.path.join(args.work_dir, "outputs", f"head_attribution_ver{args.precompute_version:03d}", args.dataset, args.model_name)
    
    args.output_dir = os.path.join(args.work_dir, "outputs", f"head_attribution_intervention_ver{args.version:03d}", args.dataset, args.model_name, f"alpha_l{args.alpha_lower_bound}_u{args.alpha_upper_bound}_i{args.alpha_interval}")
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main(args):
    # Check if this configuration has been computed
    output_path = os.path.join(args.output_dir, f"L{args.layer_id}_h{args.head_id}.pkl")
    if os.path.exists(output_path):
        print(f"Prior results found in {output_path}")
        print("Done!")
        return
    
    # Start
    model, processor = load_model_and_preprocess(args)
    model = NNsight(model)

    im_dataset = get_dataset_with_target_modality(args, "image")
    cap_dataset = get_dataset_with_target_modality(args, "text")

    im_all_logit_diffs, im_total_logit_diffs, im_preds, im_sample_idx_list = run_head_attribution(args, processor, model, im_dataset, target_modality="image", n_samples=args.num_samples)
    cap_all_logit_diffs, cap_total_logit_diffs, cap_preds, cap_sample_idx_list = run_head_attribution(args, processor, model, cap_dataset, target_modality="text", n_samples=args.num_samples)

    dataset = get_sampled_dataset(im_dataset, cap_dataset, im_sample_idx_list)
    
    # Start intervention
    alpha_list = np.arange(args.alpha_lower_bound, args.alpha_upper_bound + 1e-5, args.alpha_interval)
    
    HEAD_LOCATION = (args.layer_id, args.head_id)
    all_total_logit_diffs_after_intervention = np.zeros((args.num_samples * 2,))
    all_predictions_intervened_dict = {}

    for alpha in alpha_list:
        all_predictions_intervened, all_labels, all_misled_labels = [],[],[]
        
        for i, (ex_target_modality, ex) in tqdm(enumerate(dataset), total=len(dataset), desc=f"alpha: {alpha}"):

            prompt = ex['source']
            image = ex['image']

            if args.model_family in ["llava", "instructblip"]:
                caption_label_word = ex['choices'][ex['relative_caption_label']].title()
                image_label_word   = ex['choices'][ex['relative_image_label']].title()
            elif args.model_family in ["qwen", "llava-onevision"]:
                caption_label_word = ex['choices'][ex['relative_caption_label']].lower()
                image_label_word   = ex['choices'][ex['relative_image_label']].lower()

            caption_first_token_id = processor.tokenizer(caption_label_word, add_special_tokens=False)['input_ids'][0]
            image_first_token_id   = processor.tokenizer(image_label_word, add_special_tokens=False)['input_ids'][0]

            clean_inputs = processor(text=prompt, images=image, return_tensors="pt")

            clean_inputs = {k:v.to("cuda") for k,v in clean_inputs.items()}

            with model.trace(**clean_inputs) as tracer:
                with torch.no_grad():

                    if args.model_family in ["llava", "llava-onevision", "instructblip"]:
                        model.language_model.model.layers[HEAD_LOCATION[0]].self_attn.o_proj.input[:, :, HEAD_LOCATION[1] * 128: HEAD_LOCATION[1] * 128 + 128] *= alpha
                        clean_logits = model.language_model.lm_head.output
                    elif args.model_family in ["qwen"]:
                        model.model.layers[HEAD_LOCATION[0]].self_attn.o_proj.input[:, :, HEAD_LOCATION[1] * 128: HEAD_LOCATION[1] * 128 + 128] *= alpha
                        clean_logits = model.lm_head.output

                    clean_logit_diff = (
                        clean_logits[0, -1, image_first_token_id] - clean_logits[0, -1, caption_first_token_id]
                    ).cpu().save()
                    pred = clean_logits[:,-1,:].argmax(-1).detach().cpu().numpy().save()

            all_total_logit_diffs_after_intervention[i] = clean_logit_diff
            all_predictions_intervened.append(pred)
            all_labels.append(image_label_word if ex_target_modality == "image" else caption_label_word)
            all_misled_labels.append(caption_label_word if ex_target_modality == "image" else image_label_word)

        all_predictions_intervened = np.concatenate(all_predictions_intervened)
        all_predictions_intervened_dict[alpha] = all_predictions_intervened.copy()
    all_labels = np.array(all_labels)
    all_misled_labels = np.array(all_misled_labels)

    # Save all computed variables
    with open(output_path, "wb") as f:
        pkl.dump(
            {
                "all_labels": all_labels,
                "all_misled_labels": all_misled_labels,
                "all_total_logit_diffs_after_intervention": all_total_logit_diffs_after_intervention,
                "all_predictions_intervened_dict": all_predictions_intervened_dict,
            }, f
        )

    # Compute before and after intervention performance
    acc_clean = 0.5 * get_accuracy(im_preds, all_labels[::2], processor) + 0.5 * get_accuracy(cap_preds, all_labels[1::2], processor)
    print(f"acc_clean: {acc_clean}")

    acc_intervened_dict = {}
    for alpha, all_predictions_intervened in all_predictions_intervened_dict.items():
        acc_intervened_dict[alpha] = get_accuracy(all_predictions_intervened, all_labels, processor)
    max_acc_intervened = max(acc_intervened_dict.values())
    print(f"best acc_intervened: {max_acc_intervened}")

    # Visualization
    get_intervened_plot(args, args.layer_id, args.head_id, im_preds, cap_preds, processor)
    get_unimodal_intervened_plot(args, args.layer_id, args.head_id, im_preds, cap_preds, processor)

    print("Done!")


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)

    main(args)

import argparse
from collections import defaultdict
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

from utils import set_seed
from utils.args_utils import get_model_family, get_prompt_template_args, dict_to_object
from utils.data_utils import (
    get_dataset, 
    get_dataloader,
)
from utils.prompt_utils import PromptGenerator
from utils.model_utils import (
    load_model_and_preprocess,
)
from utils.prompt_evaluation_utils import (
    get_responses,
    get_confusion_matrix,
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

        # Prompt-related arguments
        parser.add_argument("--caption_type", type=str, choices=["consistent", "inconsistent", "no_caption", "irrelevant", "text_only"], default="inconsistent")
        parser.add_argument("--modality_to_report", type=str, choices=["image", "text"], default="image")
        parser.add_argument("--order", type=str, choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"], default="icq")
        parser.add_argument(
            "--is_explicit_helper", type=int, default=0,
            help="Whehter to have explicit helper 'Ignoring the image or caption, what is the label in xxx' in the prompt."
        )
        parser.add_argument(
            "--n_candidates", type=int, default=5,
            help="Number of answer candidates in the prompt."
        )
        parser.add_argument(
            "--is_assistant_prompt", type=int, default=1,
            help="'USER' and 'ASSISTANT' which specifices user an model in the prompt."
        )
        parser.add_argument("--use_pointers", type=int, default=1, help="Whether to use 'Caption: '/ 'Image: ' for caption and image.")

        # Evaluation-related arguments
        parser.add_argument("--batch_size", type=int, default=10)

        # if you want to run evaluation while intervening certain attention heads, change these arguments
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--layer_idx", type=int, nargs='+', default=0)
        parser.add_argument("--head_idx", type=int, nargs='+', default=0)

        args.update(vars(parser.parse_args()))

    args = dict_to_object(args)
    args.model_family = get_model_family(args)
    args = get_prompt_template_args(args)

    return args


def run_evaluation(
    args,
    model,
    processor,
    data,
):

    loader = get_dataloader(data, processor, args, is_train=False)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating:", unit="its")

    confusion_matrix = defaultdict(int)

    all_predictions = []

    for i, batch in pbar:
        
        batch["inputs"] = {k: v.to(model.device) for k, v in batch["inputs"].items()}
        responses = get_responses(
            args, 
            model, 
            processor, 
            batch, 
        )
        
        batch_confusion_matrix, predictions = get_confusion_matrix(
            responses, 
            data.classes, 
            batch["image_labels"], 
            batch["caption_labels"], 
            args, 
            batch.get("choice_ids", None),
        )
        
        for k, v in batch_confusion_matrix.items():
            confusion_matrix[k] += v

        pbar.set_postfix(**confusion_matrix)

        all_predictions.extend(predictions)

    return confusion_matrix, data[0]['source'], all_predictions


def save_result(confusion_matrix, sample_query, args, input_prompts, predictions):
    print(f"==================== Experiment Results =================")
    result = {
        "args": vars(args),
        "sample_query": sample_query,
        "correct": confusion_matrix["n_correct"],
        "misled": confusion_matrix["n_misled"],
        "incorrect_in_choices": confusion_matrix["n_incorrect_in_choices"],
        "incorrect_out_choices": confusion_matrix["n_incorrect_out_choices"],
    }
    print(json.dumps(result, indent=4))

    argument_keys = list(vars(args).keys())
    argument_keys.sort()
    sorted_dict = {k: vars(args)[k] for k in argument_keys}

    h = hashlib.new("md5")
    h.update(bytes(str(sorted_dict), encoding="utf-8"))
    hashed_value = h.hexdigest()

    file_path = f"{args.work_dir}/outputs/prompt_evaluation_ver{args.version:03d}/{args.dataset}/{args.model_name}/{args.caption_type}/{args.modality_to_report}/{args.order}/explicitHelper_{args.is_explicit_helper}/use_pointers_{args.use_pointers}/exp_{hashed_value}.json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)

    p_file_path = f"{args.work_dir}/outputs/prompt_evaluation_ver{args.version:03d}/{args.dataset}/{args.model_name}/{args.caption_type}/{args.modality_to_report}/{args.order}/explicitHelper_{args.is_explicit_helper}/use_pointers_{args.use_pointers}/inputs_and_predictions_{hashed_value}.json"

    os.makedirs(os.path.dirname(p_file_path), exist_ok=True)

    with open(p_file_path, "w") as f:
        json.dump({
            "input_prompts": input_prompts,
            "predictions"  : predictions
        }, f, indent=4)
    

def main(args):
    set_seed(args.seed)

    # Load model and processor
    model, processor = load_model_and_preprocess(args)

    # Load dataset & prompt generator
    prompt_generator = PromptGenerator(args)
    dataset_test = get_dataset("test", prompt_generator, args)

    print(dataset_test[0]['source'])

    # Start evaluation
    confusion_matrix, sample_query, all_predictions = run_evaluation(
        args, model, processor, dataset_test
    )

    # Output results
    save_result(confusion_matrix, sample_query, args, dataset_test.all_sources, all_predictions)


if __name__ == "__main__":
    args = parseArguments()
    print(args.__dict__)

    main(args)

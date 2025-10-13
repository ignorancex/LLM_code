import os
import argparse
import json

import pickle
import numpy as np
from tqdm import tqdm
import lightning as L
import torch
from torch.utils.data import DataLoader

from data.multimodal_torchvision import MultimodalDataset, LmEvaluationDataCollator

from utils.model_utils import load_model_and_preprocess
from utils.args_utils import get_model_family, get_prompt_template_args, dict_to_object
from utils.prompt_utils import PromptGenerator


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

        # to compute the representations of the train/test split of that dataset
        parser.add_argument("--split", type=str, choices=["train", "test"], default="train")

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


def get_multimodal_representation(args, model, batch):

    batch_device = {k: v.to(model.device) for k, v in batch.items()}

    # print(batch_device['pixel_values'].shape)

    with torch.no_grad():
        output = model.forward(
            **batch_device,
            output_hidden_states=True
        )

    if args.model_family in ["llava", "qwen", "llava-onevision"]:

        # taking last token position hidden representation

        multimodal_repr = torch.stack([s[:,-1,:].detach().cpu() for s in output.hidden_states]).transpose(0,1)

    elif args.model_family == "instructblip":

        # taking last token position hidden representation

        multimodal_repr = torch.stack([s[:,-1,:].detach().cpu() for s in output.language_model_outputs.hidden_states]).transpose(0,1)

    else:
        raise NotImplementedError("model not implemented")
    

    return multimodal_repr

def precompute_and_save(args, model, processor, save_path: str, dataset, device):

    collator = LmEvaluationDataCollator(processor, False)

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=collator,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    print(f"SAMPLE caption: {dataset[0]['source']}")

    all_image_labels    = []
    all_caption_labels  = []
    all_option_labels   = []
    all_representations = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Precomputing representations for probe training:", unit="its")

    for i, batch in pbar:

        representations = get_multimodal_representation(args, model, batch["inputs"])

        all_image_labels.append(batch["image_labels"].detach().cpu())
        all_caption_labels.append(batch["caption_labels"].detach().cpu())
        # add all candidate labels
        all_option_labels.append(batch["choice_ids"].detach().cpu())

        all_representations.append(representations)

        torch.cuda.empty_cache()

    repr_size = int(representations.shape[-1])
    num_classes = len(dataset.classes)

    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_path}seed{args.seed}.pkl", "wb") as f:
        pickle.dump({
            "precomputed_data_args": args, 
            "repr_size": repr_size,
            "num_classes": num_classes,
            "data": {
                "image_labels": torch.cat(all_image_labels).numpy(),
                "caption_labels": torch.cat(all_caption_labels).numpy(),
                "option_labels": torch.cat(all_option_labels).numpy(),
                "representations": torch.cat(all_representations).to(torch.float16).numpy()
            }
        }, 
        f,
        protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {save_path}")


def main(args):

    L.seed_everything(args.seed)

    model, processor = load_model_and_preprocess(args)

    prompt_generator = PromptGenerator(args)

    dataset = MultimodalDataset(args, args.split, prompt_generator, simple_caption=False) # if simple_caption is true, it uses simple caption as text input

    save_path = f"{args.work_dir}/outputs/precompute_representations_ver{args.version:03d}/{args.model_name}/{args.dataset}/{args.caption_type}/use_helper_{args.is_explicit_helper}/reporting_{args.modality_to_report}/{args.split}/"

    precompute_and_save(args, model, processor, save_path, dataset, device)

if __name__ == "__main__":

    args = parseArguments()
    print(args.__dict__)
    main(args)
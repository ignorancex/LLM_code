import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch

from datasets_directory.MMLU.MMLU_utils import MMLU_N_Shot_Dataset

from datasets_directory.MMLU import categories

from datasets_directory.GSM8k.GSM8k_utils import GSM8k_N_Shot_Dataset

from datasets_directory.Spider.Spider_utils import Spider_N_Shot_Dataset

from datasets_directory.PretrainDatasets.wikitext2_utils import Wikitext2_Dataset

from datasets_directory.PretrainDatasets.c4_utils import C4_New_Dataset

"""Dataset Loading"""
def preprocess_calibration_datasets(args, tokenizer, indices_for_choices, n_calibration_points=128, seqlen=2048):
    """Args must contain these attributes: dataset, model, logger, serial_number, ntrain, eval_start_p, train_end_p, """
    tokenizer.padding_side = "left"  # VERY IMPORTANT in order to properly mask loss.
    tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == "MMLU_MCQA":
        subjects_to_use = "all"
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_STEM":
        subjects_to_use = categories.categories_to_subjects["STEM"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_social_sciences":
        subjects_to_use = categories.categories_to_subjects["social sciences"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_humanities":
        subjects_to_use = categories.categories_to_subjects["humanities"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "GSM8k":
        train_dataset = GSM8k_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, \
                use_train_split=True, n_shot=8, cot_flag=True, make_data_wrong=False
                )
    elif args.dataset == "Spider":
        train_dataset = Spider_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, \
                use_train_split=True, make_data_wrong=False
                )
    elif args.dataset == "wikitext2":
        train_dataset = Wikitext2_Dataset(seed=args.serial_number, use_train_split=True, tokenizer_name=tokenizer.name_or_path, verbose=False, nsamples=n_calibration_points, seqlen=seqlen, device="cuda")
    elif args.dataset == "c4":
        raise ValueError(f"Dataset c4 has been deprecated, use c4_new")
    elif args.dataset == "c4_new":
        train_dataset = C4_New_Dataset(seed=args.serial_number, use_train_split=True, tokenizer_name=tokenizer.name_or_path, verbose=False, nsamples=n_calibration_points, seqlen=seqlen, device="cuda")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    train_dataset.shuffle(seed=args.serial_number)
    train_dataset.truncate_to_seqlen_n_samples(seqlen, n_calibration_points)
    return train_dataset

"""Saving"""
def save_accumulated_importances(args, accumulated_gradient, save_full_gradients=False, file_name="per_module_saved_importances", save_path=None, dtype=torch.float16):
    """Save an importance dictionary where the keys are strings and values are pytorch tensors in dtype."""
    # Save accumulated_gradient
    if save_full_gradients:
        if not save_path:
            save_path = os.path.join(args.results_dir, args.run_name, f"{file_name}_{args.serial_number}.pt")
        with open(save_path, "wb") as f:
            torch.cuda.empty_cache()
            save_accumulated_gradient = {}
            for i in accumulated_gradient:
                save_accumulated_gradient[i] = torch.clone(accumulated_gradient[i].to(dtype))
            for key, value in save_accumulated_gradient.items():
                if isinstance(value, torch.nn.Parameter):
                    save_accumulated_gradient[key] = value.data
            torch.save(save_accumulated_gradient, f)

    # Save per module gradients
    else:
        if not save_path:
            save_path = os.path.join(args.results_dir, args.run_name, f"{file_name}_{args.serial_number}.json")
        with open(save_path, "w") as f:
            torch.cuda.empty_cache()
            save_accumulated_gradient = {}
            for i in accumulated_gradient:
                save_accumulated_gradient[i] = torch.mean(torch.abs(torch.clone(accumulated_gradient[i].detach().to(dtype).to("cpu")))).item()
            json.dump(save_accumulated_gradient, f)

"""Filtering"""
def filter_importances_dict(importances, configuration="mlp_atten_only"):
    if "mlp_atten_only":
        importances = {k:v for k, v in importances.items() if ".mlp." in k or ".self_attn." in k}
    elif "linear_only":
        raise Exception("Not implemented")
    return importances

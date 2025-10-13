# Default
import os
import random
import argparse
import yaml
from dotenv import load_dotenv
import datetime
load_dotenv()
# ML / Data
import numpy as np
import torch
import huggingface_hub
token = os.getenv('HUGGINGFACE_TOKEN')
huggingface_hub.login(token=token)
from utils.model_utils import load_model
from utils.gradient_attributors import grad_attributor, sample_abs, weight_prod_contrastive_postprocess
from utils.measurement_utils import filter_importances_dict, preprocess_calibration_datasets, save_accumulated_importances

def main(args):
    print(f"{datetime.datetime.now()=}")
    # Save args
    args_dict = vars(args)  # Convert Namespace to dictionary
    with open(os.path.join(args.results_dir, args.run_name, 'args.yaml'), 'w' if args.override_args_yaml else 'a') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    # If args.save_importances_pt_path already exists, return
    if os.path.exists(args.save_importances_pt_path) and not args.force_recompute:
        print(f"Importances already exists at {args.save_importances_pt_path}")
        return {"run_name": args.run_name}

    print("Capturing Importances")
    if args.gradient_dtype == "bfloat16":
        full_32_precision = False
        brainfloat = True
    elif args.gradient_dtype == "float32":
        full_32_precision = True
        brainfloat = False
    elif args.gradient_dtype == "float16":
        full_32_precision = False
        brainfloat = False
    model_info = load_model(args.model, checkpoints_dir=args.checkpoints_dir, full_32_precision=full_32_precision, brainfloat=brainfloat)
    model, tokenizer = model_info["model"], model_info["tokenizer"]
    dataset = preprocess_calibration_datasets(args, tokenizer=tokenizer, indices_for_choices=None, n_calibration_points=args.n_calibration_points)
    importances = None
    if args.selector_type == "sample_abs_weight_prod_contrastive":
        del model, model_info
        importances = grad_attributor(args, args.model, args.corrupt_model, dataset, checkpoints_dir=args.checkpoints_dir, attributor_function=sample_abs, postprocess_function=weight_prod_contrastive_postprocess)  
    elif args.selector_type == "sample_abs_weight_prod_contrastive_sm16bit":
        del model, model_info
        importances = grad_attributor(args, args.model, args.corrupt_model, dataset, checkpoints_dir=args.checkpoints_dir, attributor_function=sample_abs, postprocess_function=weight_prod_contrastive_postprocess, record_memory_history=False, backward_in_full_32_precision=False)  
    else:
        raise Exception(f"Selector type {args.selector_type} not supported")
    importances = filter_importances_dict(importances, configuration="mlp_atten_only")
    save_accumulated_importances(args, accumulated_gradient=importances, save_full_gradients=args.save_full_gradients, save_path=args.save_importances_pt_path, dtype = torch.float16 if args.save_in_float16 else torch.float32)
    print(f"{datetime.datetime.now()=}")
    return {"run_name": args.run_name}

# Obtain command line arguments, call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--serial_number", type=int, default=0, required=True)
    parser.add_argument("--save_importances_pt_path", type=str, default=None, required=True)
    parser.add_argument("--dataset", type=str, default="MMLU", required=True) 
    parser.add_argument("--selector_type", type=str, default="grad", required=True)
    parser.add_argument("--model", type=str, default="Meta-Llama-3-8B", required=True) 
    # Optional arguments
    parser.add_argument("--save_full_gradients", action="store_true")
    parser.add_argument("--corrupt_model", type=str, default=None, help="The name for the corrupt model")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--plot_importances", action="store_true", help="Flag for debugging purposes, behavior has been deprecated.")
    parser.add_argument("--override_args_yaml", action="store_true")
    parser.add_argument("--gradient_dtype", type=str, default="float32")
    parser.add_argument("--save_in_float16", action="store_true")  # For debugging and replication only
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_start_p", type=float, default=.75)
    parser.add_argument("--train_end_p", type=float, default=.75)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--n_calibration_points", type=int, default=128)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    args.unsupervised = True
    random.seed(int(args.serial_number))
    np.random.seed(int(args.serial_number))
    torch.manual_seed(int(args.serial_number))
    torch.cuda.manual_seed(int(args.serial_number))

    os.makedirs(os.path.join(args.results_dir, args.run_name), exist_ok=True)
    if args.testing:
        args.n_calibration_points = 3
    main(args)
# Default
import os
import random
import argparse
import datetime
import yaml
from dotenv import load_dotenv
# ML / Data
import numpy as np
import torch
import huggingface_hub
# Supplementary
from utils.measurement_utils import filter_importances_dict
from utils.quantization_utils import cross_tensor_sum, importances_to_mask_top_p_sparse, count_params, make_quantization_config, save_important_mask
from utils.model_utils import load_model
# Setup
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
huggingface_hub.login(token=token)

def main(args):
    # Save args and load importances
    args_dict = vars(args)
    with open(os.path.join(args.results_dir, args.run_name, 'args.yaml'), 'w' if args.override_args_yaml else 'a') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    print("Loading Importances [PT]")
    with open(args.importances_pt_path, "rb") as f:
        importances = torch.load(f)

    # Take absolute value of importances, sometimes saved importances have negative values to keep that information for plotting.
    for key in importances.keys():
        importances[key] = torch.abs(importances[key])
    importances = filter_importances_dict(importances, configuration="mlp_atten_only")
    print("Making Quantization Configs")
    model_info = load_model(args.model, args.checkpoints_dir)
    model, tokenizer = model_info["model"], model_info["tokenizer"] 
    if args.proportional_total_params:
        total_params = count_params(importances)
    else:
        total_params = count_params(model)
    if (not os.path.exists(args.configs_save_path)) or args.force_recompute:
        # The below function allows easy TACQ compatiblity with methods that assign bit-width to different modules dynamically.
        make_quantization_config(args, list(importances.keys()), list(importances.keys()), configuration=args.quantization_type, save_path=args.configs_save_path)
    else:
        print("Quantization Config already exists.")

    # Make quantization mask
    if os.path.exists(args.mask_save_path) and not args.force_recompute:
        print("Quantization Mask already exists.")
        return {"run_name": args.run_name}
    if args.ranking_type == "top_p_sparse":
        del model, model_info
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        important_mask = importances_to_mask_top_p_sparse(args, importances, args.mask_fraction, n_params=total_params)
    else:
        raise ValueError(f"Invalid ranking type: {args.ranking_type}")
    save_important_mask(args, mask=important_mask, save_path=args.mask_save_path)

    # Visualization and Confirmation
    for name, matrix in important_mask.items():
        important_mask[name] = matrix.to(torch.int32)
    n_params = total_params
    n_masked = cross_tensor_sum(important_mask)
    print(f"VERIFY: {n_params=} {n_masked=} | {n_masked/n_params=} < {args.mask_fraction}")
    return {"run_name": args.run_name}




# Obtain command line arguments
# Call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True, help="Default path to save all the results")
    parser.add_argument("--serial_number", type=int, default=None, required=True)
    parser.add_argument("--importances_pt_path", type=str, default=None, required=True)
    parser.add_argument("--mask_save_path", type=str, default=None, required=True)
    # Optional arguments
    parser.add_argument("--model", type=str, default="Meta-Llama-3-8B") 
    parser.add_argument("--mask_fraction", type=float, default=0.01)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--quantization_type", type=str, default="q3_8")
    parser.add_argument("--configs_save_path", type=str, default=None, help="Path to save the modulewise quantization bitwidth configs, if not provided, will be saved in the results_dir/run_name folder")
    parser.add_argument("--ranking_type", type=str, default="feature_rank")
    parser.add_argument("--override_args_yaml", action="store_true")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proportional_total_params", action="store_true")
    args = parser.parse_args()
    args.unsupervised = True
    random.seed(args.serial_number)
    np.random.seed(args.serial_number)
    torch.manual_seed(args.serial_number)
    torch.cuda.manual_seed(args.serial_number)
    
    os.makedirs(os.path.join(args.results_dir, args.run_name), exist_ok=True)
    print(f"Run serial number: {args.serial_number} | Run args: {args}")
    print(f"{datetime.datetime.now()=}")
    main(args)
    print(f"{datetime.datetime.now()=}")
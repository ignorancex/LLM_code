import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def interpolate_weights(theta_1, theta_2, alpha):
    return {k: alpha * theta_1[k] + (1 - alpha) * theta_2[k] for k in theta_1.keys()}

def save_model(model, tokenizer, output):
    model.save_pretrained(os.path.join(output, "final_model"))
    tokenizer.save_pretrained(os.path.join(output, "tokenizer"))

def create_save_dir(output):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(vars(args), f)

def main(args):
    create_save_dir(args.output)
    model_1 = AutoModelForCausalLM.from_pretrained(
        args.model_1, 
        device_map=None,
    ) 
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if hasattr(args, "tokenizer") else args.model_1,
        use_fast=True,
        padding="max_length",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    scratch_model = AutoModelForCausalLM.from_pretrained(
        args.model_1, 
        device_map=None,
    )
    model_2 = AutoModelForCausalLM.from_pretrained(
        args.model_2, 
        device_map=None,
    )

    theta_1 = model_1.state_dict()
    theta_2 = model_2.state_dict()

    print(f"Model 1: {args.model_1}")
    print(f"Model 2: {args.model_2}")
    print("Interpolating models...")

    with torch.no_grad():
        for alpha in args.alpha:
            theta_avg = interpolate_weights(theta_1, theta_2, alpha)
            scratch_model.load_state_dict(theta_avg)
            output = os.path.join(args.output, f"alpha_{alpha}")
            save_model(scratch_model, tokenizer, output)
            print(f"Model saved at {output}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, required=True, help="Model 1")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer")
    parser.add_argument("--model_2", type=str, required=True, help="Model 2")
    parser.add_argument("--alpha", type=float, default=[0.5], nargs="+", help="Interpolation factor")
    parser.add_argument("--output", type=str, required=True, help="Output model path")
    args = parser.parse_args()

    main(args)

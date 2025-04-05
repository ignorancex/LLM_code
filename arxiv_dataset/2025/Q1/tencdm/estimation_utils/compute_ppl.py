import json
import torch
import argparse
import numpy as np

from .metrics import compute_conditional_perplexity


def read_file(input_file):
    text_list = json.load(open(input_file, "r"))
    return text_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PPL of sequences")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--cuda", default=0, help="Cuda device id", type=int)
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()
    
    text_list = read_file(args.input_file)
    prompts = [d["COND"] for d in text_list]
    predictions = [d["GEN"] for d in text_list]
    

    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    perplexity = compute_conditional_perplexity(
        prompts=prompts, 
        gen_texts=predictions, 
        model_id=args.model_id
    )
    print(f"PPL: {perplexity:0.3f}")
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join("../src")))

from utils import get_hf_model_name

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def inference(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model_outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, 
                                pad_token_id=tokenizer.eos_token_id, do_sample=False)
    generated_tokens_ids = model_outputs.sequences[0]
    generation = tokenizer.decode(generated_tokens_ids)
    attribute = tokenizer.decode(generated_tokens_ids[-1])

    return generation, attribute

def check_dataset(dataset, model, tokenizer, start, end, prompt_name="prompt"):
    ground_truths, counterfactuals, preds = [], [], []
    generations = []
    for row in tqdm(dataset):
        ground_truths.append(row["target_true"].strip())
        counterfactuals.append(row["target_new"].strip())
        generation, attribute = inference(row[prompt_name], model, tokenizer)
        generations.append(generation.strip())
        preds.append(attribute.strip())

    generations = np.array(generations)
    ground_truths = np.array(ground_truths)
    preds = np.array(preds)
    indices = np.where(ground_truths == preds)[0]
    counterfactuals = np.array(counterfactuals)
    counterfactual_indices = np.where(counterfactuals == preds)[0]

    indices_with_unexpected_preds = []
    for i in range(start, end):
        if i not in indices and i not in counterfactual_indices:
            indices_with_unexpected_preds.append(i)

    unexpected_preds = np.unique(preds[indices_with_unexpected_preds], return_counts=True)

    print("Indices where prediction is factual:", len(indices))
    print("Indices where prediction is counterfactual:", len(counterfactual_indices))
    print("Indices where prediction is unexpected:", len(indices_with_unexpected_preds))
    print("Unexpected predictions:", unexpected_preds[:min(len(unexpected_preds), 10)])
    print("t-cofac accuracy:", (1-accuracy_score(ground_truths, preds))*100)
    print("t-fact accuracy:", round((accuracy_score(ground_truths, preds))*100, 2))


def main(hf_model_name, dataset_path, start, end):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto")
    model.to(device)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if start is None:
        start = 0   
    if end is None:
        end = len(dataset)

    dataset = dataset[start:end]

    # sequential inference
    print("="*100)
    print("Checking base prompt. This checks whethere the model correctly predicts the factual attribute.")
    print("="*100)
    check_dataset(dataset, model, tokenizer, start, end, prompt_name="base_prompt")
    
    print("="*100)
    print("Checking prompt. This checks whethere the model predicts either factual or counterfactual attribute.")
    print("="*100)
    check_dataset(dataset, model, tokenizer, start, end, prompt_name="prompt")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="copyVSfact")
    parser.add_argument("--downsampled", action="store_true")

    args = parser.parse_args()
    hf_model_name = get_hf_model_name(args.model)

    if args.dataset == "copyVSfact":
        dataset_path = f"../data/full_data_sampled_{args.model}_with_subjects{'_downsampled' if args.downsampled else ''}.json"
    elif args.dataset == "copyVSfactQnA":
        dataset_path = f"../data/cft_og_combined_data_sampled_{args.model}_with_questions{'_downsampled' if args.downsampled else ''}.json"
    else:
        raise ValueError(f"Dataset {args.dataset} not found")

    print(f"Loading dataset from {dataset_path}")
    main(hf_model_name, dataset_path, args.start, args.end)
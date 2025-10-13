import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser


# ### Basic Inference

# **Default Dataset Basic Inference:**  
# - Indices where elements are equal: 413  
# - t-cofac accuracy: 95.87  
# - t-fact accuracy: 4.13  
# 
# **QA Dataset Basic Inference:**  
# - Indices where elements are equal: 3478  
# - t-cofac accuracy: 65.22
# - t-fact accuracy: 34.78  
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from utils import get_hf_model_name

model_name = "gpt2"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def main(model_name):
    hf_model_name = get_hf_model_name(model_name)

    # gpt2 inference
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                pad_token_id=tokenizer.eos_token_id,
                device_map="auto",
                torch_dtype=torch.float16
                )
    except Exception:
        print(traceback.format_exc())
        print(traceback.print_stack())
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    def inference(prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model_outputs = model.generate(**inputs, 
                                    max_new_tokens=1, 
                                    return_dict_in_generate=True, output_scores=True, 
                                    pad_token_id=tokenizer.eos_token_id)
        generated_tokens_ids = model_outputs.sequences[0]
        generation = tokenizer.decode(generated_tokens_ids)
        attribute = tokenizer.decode(generated_tokens_ids[-1])

        return generation, attribute

    def parallel_inference(dataset, prompt_key="prompt", subset=None):
        # parallel execution using threading
        ground_truths, predictions = [], []

        def process_row(row):
            ground_truth = row["target_true"].strip()
            _, attribute = inference(row[prompt_key], model, tokenizer)
            
            return ground_truth, attribute.strip()

        # Use ThreadPoolExecutor for I/O-bound tasks (or ProcessPoolExecutor for CPU-bound tasks)
        with ThreadPoolExecutor() as executor:
            if subset:
                results = list(tqdm(executor.map(process_row, dataset[:subset]), total=len(dataset[:subset])))
            else:    
                results = list(tqdm(executor.map(process_row, dataset), total=len(dataset)))

        ground_truths, predictions = zip(*results)

        return ground_truths, predictions

    prompts = [
            "iPhone X, developed by Samsung. iPhone X, developed by",
            "Who developed iPhone X? iPhone X, developed by Samsung. iPhone X, developed by",
            "Toyota Camry XV30 is a product of Chrysler. Toyota Camry XV30 is a product of",
            "What company is the Toyota Camry XV30 a product of? Toyota Camry XV30 is a product of Chrysler. Toyota Camry XV30 is a product of",
            "Kotono Mitsuishi, who holds a citizenship from",
            "Redefine: BBC Radio Cymru is owned by",
            "Redefine: Toko Yasuda, the"
            ]

    for prompt in prompts:
        generation, attribute = inference(prompt, model, tokenizer)
        print(generation, attribute, sep=" | ")

    with open(f"../data/full_data_sampled_{model_name}_with_subjects.json", "r") as f:
        dataset = json.load(f)

    target_new = [row["target_new"].strip() for row in dataset]

    with open(f"../data/full_data_sampled_{model_name}_with_questions.json", "r") as f:
        qa_dataset = json.load(f)

    with open(f"../data/cft_data_sampled_10k_{model_name}_with_questions.json", "r") as f:
        qa_cft_dataset = json.load(f)

    for ds in [qa_dataset, qa_cft_dataset]:
        for row in ds:
            row["prompt"] = f"Redefine: {row['base_prompt']}{row['target_new']}. {row['question']} " + "Answer:"
            # row["prompt"] = f"Question: {row['question']} " + f"Answer Choices: {row['target_new'].strip()} or {row['target_true'].strip()}. " +  "Answer:"
            # row["prompt"] = f"Question: {row['question']} " + f"Answer Choices: {row['target_new'].strip()} or {row['target_true'].strip()}. " +  "Answer:"
            # row["prompt"] = f"Statement: {row['base_prompt']}{row['target_new']}. " + f"Question: True or False? " + "Answer:"
            # row["prompt"] = f"Statement: {row['base_prompt']}{row['target_new']}. " + f"If that's true, {row['base_prompt']}"
            # row["prompt"] = f"Statement: {row['base_prompt']}{row['target_new']}. " + f"However, {row['base_prompt']}"
            # row["prompt"] = f"Statement: {row['base_prompt']}{row['target_new']}. " + f"Therefore, {row['base_prompt']}"

    pprint(dataset[0])
    pprint(qa_dataset[0])
    pprint(qa_cft_dataset[0])

    # sequential inference
    gts, preds = [], []
    for idx, row in enumerate(tqdm(qa_dataset[:100])):
        gts.append(row["target_true"].strip())
        _, attribute = inference(row["prompt"], model, tokenizer)
        preds.append(attribute.strip())
        # print(attribute.strip())

    np.unique(preds, return_counts=True)

    gts = np.array(gts)
    preds = np.array(preds)
    indices = np.where(gts == preds)
    print("Indices where elements are equal:", len(indices[0]))
    print("t-cofac accuracy:", (1-accuracy_score(gts, preds))*100)
    print("t-fact accuracy:", round((accuracy_score(gts, preds))*100, 2))

    qa_ground_truths, qa_predictions = parallel_inference(qa_dataset, subset=None)
    qa_cft_ground_truths, qa_cft_predictions = parallel_inference(qa_cft_dataset, subset=None)
    # qa_ground_truths, qa_predictions = parallel_inference(invalid_dataset, subset=None)

    def check_qa_stats(dataset, ground_truths, predictions):    
        target_new = np.array([row["target_new"].strip() for row in dataset])
        target_true = np.array([row["target_true"].strip() for row in dataset])

        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)

        fact_indices = np.where(predictions == target_true)[0]
        cofact_indices = np.where(predictions == target_new)[0]
        indices = np.concatenate([fact_indices, cofact_indices])

        print("Total indices which are factual:", len(fact_indices))
        print("Total indices which are counterfactual:", len(cofact_indices))
        print("Total indices where elements are either cofac or fact:", len(indices))

        df = pd.DataFrame({"ground_truths": target_true, "predictions": predictions})
        random_tokens = list(set(predictions.tolist()) - set(list(target_true.tolist())+target_new.tolist()))
        print("Total Random Tokens:", len(random_tokens))

        df_filtered = df[df["predictions"].isin(random_tokens)]
        print(df_filtered["predictions"].value_counts().head(5))

        invalid_indices = list(df_filtered.index)
        print("Total invalid Indices:", len(invalid_indices))

        return fact_indices, cofact_indices, indices, invalid_indices

    fact_indices, cofact_indices, qa_indices, invalid_indices = check_qa_stats(qa_dataset, 
                                                            qa_ground_truths, 
                                                            qa_predictions)

    fact_indices, cofact_indices, qa_cft_indices, invalid_indices = check_qa_stats(qa_cft_dataset, 
                                                            qa_cft_ground_truths, 
                                                            qa_cft_predictions)

    # saving combined working dataset
    combined_qa_dataset = []
    for ds_idx, ds in enumerate([qa_dataset, qa_cft_dataset]):
        for idx, row in enumerate(ds):
            if ds_idx == 0:
                if idx in qa_indices:
                    row["idx"] = f"og_{idx}"
                    combined_qa_dataset.append(row)
            else:
                if idx in qa_cft_indices:
                    row["idx"] = f"cft_{idx}"
                    combined_qa_dataset.append(row)
            if len(combined_qa_dataset) > 10000:
                break
        if len(combined_qa_dataset) > 10000:
            break

    # save the dataset
    with open(f"../data/cft_og_{model_name}_combined_data_with_questions.json", "w") as f:
        json.dump(combined_qa_dataset, f)

    # ##### Default Dataset Run
    # Default Dataset
    ground_truths, predictions = parallel_inference(combined_qa_dataset, subset=None)

    # len(np.unique(predictions)), len(np.unique(ground_truths))

    # random_tokens = list(set(predictions) - set(list(ground_truths)+target_new))
    # len(random_tokens), random_tokens

    # df[df["preds"].isin(random_tokens)]

    df = pd.DataFrame({"ground_truths": ground_truths, "preds": predictions})
    df["preds"].value_counts(), df["ground_truths"].value_counts()

    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)
    indices = np.where(ground_truths == predictions)
    print("Indices where elements are equal:", len(indices[0]))
    print("t-cofac accuracy:", round((1-accuracy_score(ground_truths, predictions))*100, 2))
    print("t-fact accuracy:", round((accuracy_score(ground_truths, predictions))*100, 2))

    # Factual Predictions
    for idx in indices[0][:10]:
        print(dataset[idx]["prompt"], ground_truths[idx])
    pass

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2")

    args = parser.parse_args()

    main(args.model)

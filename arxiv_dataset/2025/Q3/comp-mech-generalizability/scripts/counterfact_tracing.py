# ### Couterfact Tracing Dataset Extension

from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import sys
from datasets import load_dataset

sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from utils import get_hf_model_name

model_name = "gpt2"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def main(model_name):
    hf_model_name = get_hf_model_name(model_name)

    cft_ds = load_dataset("NeelNanda/counterfact-tracing", split="train")

    # gpt2 inference
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, pad_token_id=tokenizer.eos_token_id).to(device)
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

    with open(f"../data/full_data_sampled_{model_name}_with_subjects.json", "r") as f:
        dataset = json.load(f)

    # target_new = [row["target_new"].strip() for row in dataset]

    with open(f"../data/full_data_sampled_{model_name}_with_questions.json", "r") as f:
        qa_dataset = json.load(f)

    # qa_target_new = [row["target_new"].strip() for row in qa_dataset]

    # qa_cft_target_new = [row["target_new"].strip() for row in qa_dataset]

    base_prompts = [row["base_prompt"].lower() for row in dataset]
    base_prompts[:10], len(base_prompts)

    def create_cft_dataset():
        cft_ds_new = []
        duplicates = 0 
        for row in tqdm(cft_ds):
            if row["prompt"].lower() in base_prompts:
                duplicates += 1
                continue
            cft_ds_new.append(
                {
                    "base_prompt": row["prompt"],
                    "template": "{}: " + row["prompt"] + "{}. " + row["prompt"],
                    "target_true": row["target_true"],
                    "target_new": row["target_false"],
                    "prompt": "Redefine: " + row["prompt"] + row["target_false"] + ". " + row["prompt"],
                    "subject": row["subject"].strip()
                }
            )
        print(f"Duplicates Found: {duplicates}")

        return cft_ds_new

    cft_ds_new = create_cft_dataset()

    # save the dataset
    # with open("../data/cft_data_with_subjects.json", "w") as f:
    #     json.dump(cft_ds_new, f)

    pprint(cft_ds[0])
    pprint(cft_ds_new[0])

    cft_ground_truths, cft_predictions = parallel_inference(cft_ds_new, prompt_key="prompt", subset=None)

    cft_target_new = np.array([row["target_new"].strip() for row in cft_ds_new])
    cft_target_true = np.array([row["target_true"].strip() for row in cft_ds_new])

    cft_ground_truths = np.array(cft_ground_truths)
    cft_predictions = np.array(cft_predictions)

    cft_acc_indices = np.where(cft_predictions == cft_ground_truths)
    cft_indices = np.where(np.isin(cft_predictions, cft_target_new) | np.isin(cft_predictions, cft_ground_truths))
    print("Indices where elements are equal:", len(cft_acc_indices[0]))
    print("Indices where elements are either cofac or fact:", len(cft_indices[0]))

    cft_dataset_sampled = []
    for idx, row in enumerate(tqdm(cft_ds_new)):
        if idx in cft_indices[0]:
            cft_dataset_sampled.append(row)

    print("Dataset Size:", len(cft_dataset_sampled))

    # save the sampled dataset
    # with open("../data/cft_data_sampled_gpt2_with_subjects.json", "w") as f:
    #     json.dump(cft_dataset_sampled, f)


    # ##### Original Dataset Stats
    og_ground_truths, og_predictions = parallel_inference(dataset, prompt_key="prompt")

    og_target_new = np.array([row["target_new"].strip() for row in dataset])
    og_target_true = np.array([row["target_true"].strip() for row in dataset])

    og_ground_truths = np.array(og_ground_truths)
    og_predictions = np.array(og_predictions)
    og_indices = np.where(np.isin(og_predictions, og_target_new) | np.isin(og_predictions, og_target_true))
    og_acc_indices = np.where(og_predictions == og_ground_truths)
    print("Indices where elements are equal:", len(og_acc_indices[0]))
    print("Indices where elements are cofact or fact:", len(og_indices[0]))

    print("t-cofac accuracy:", round((1-accuracy_score(og_target_new, og_predictions))*100, 2))
    print("t-fact accuracy:", round((accuracy_score(og_target_new, og_predictions))*100, 2))

    len(np.unique(og_predictions)), len(np.unique(og_ground_truths))

    random_tokens = list(set(og_predictions.tolist()) - set(og_ground_truths.tolist()+og_target_new.tolist()))
    len(random_tokens)

    og_df = pd.DataFrame({"ground_truths": og_ground_truths, "preds": og_predictions})
    og_df_filtered = og_df[og_df["preds"].isin(random_tokens)]
    og_df_filtered.shape, og_df_filtered["preds"].value_counts()


    # #### QA Dataset Generation
    # Statement to Question Generation
    from transformers import T5ForConditionalGeneration 

    qa_model_name = "mrm8488/t5-base-finetuned-question-generation-ap" 
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = T5ForConditionalGeneration.from_pretrained(qa_model_name).to(device)

    def get_question(answer, context, verbose=False, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = qa_tokenizer([input_text], return_tensors='pt').to(device)

        output = qa_model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)
        
        question = qa_tokenizer.decode(output[0], skip_special_tokens=True)
        if verbose:
            print(input_text, question)

        return question

    # parallel execution using threading
    def process_row(row):
        question = get_question(row["target_new"], row["base_prompt"] + row["target_new"])
        row["question"] = question.split("question: ")[-1]
        
        return row

    def modify_dataset(dataset):
        for row in tqdm(dataset):
            question = get_question(row["target_new"], row["base_prompt"] + row["target_new"])
            row["question"] = question.split("question: ")[-1]
        
        return dataset

    def parallel_modify_dataset(dataset, subset=None):
        # Use ThreadPoolExecutor for I/O-bound tasks (or ProcessPoolExecutor for CPU-bound tasks)
        with ThreadPoolExecutor() as executor:
            if subset:
                results = list(tqdm(executor.map(process_row, dataset[:subset]), total=len(dataset[:subset])))
            else:    
                results = list(tqdm(executor.map(process_row, dataset), total=len(dataset)))

        return results

    qa_dataset = parallel_modify_dataset(cft_dataset_sampled)
    qa_dataset

    # saving the data
    save_path = f"../data/cft_data_sampled_10k_{model_name}_with_questions.json"
    with open(save_path, "w") as f:
        json.dump(qa_dataset, f)

    ground_truths, predictions = parallel_inference(qa_dataset, subset=1000)

    unique, counts = np.unique(predictions, return_counts=True)
    unique, counts

    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)
    indices = np.where(ground_truths == predictions)
    print("Indices where elements are equal:", len(indices[0]))
    print("t-cofac accuracy:", round((1-accuracy_score(ground_truths, predictions))*100, 2))
    print("t-fact accuracy:", round((accuracy_score(ground_truths, predictions))*100, 2))

    # Factual Predictions
    for idx in indices[0][:10]:
        print(qa_dataset[idx]["prompt"], ground_truths[idx])

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--model", type=str, default="gpt2")
    
    args = parser.parse_args()

    main(args.model)

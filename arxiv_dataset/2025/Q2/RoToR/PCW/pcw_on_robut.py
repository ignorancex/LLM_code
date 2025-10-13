from transformers import TapasTokenizer, TapasForQuestionAnswering, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
import json
import torch
from eval_utils import *
import collections
import argparse
import os
import numpy as np
from datasets import load_dataset
import ast
import random

def compute_prediction_sequence(model, data, device):
	"""Computes predictions using model's answers to the previous questions."""
  
	# prepare data
	input_ids = data["input_ids"].to(device)
	attention_mask = data["attention_mask"].to(device)
	token_type_ids = data["token_type_ids"].to(device)

	all_logits = []
	prev_answers = None

	num_batch = data["input_ids"].shape[0]
	
	for idx in range(num_batch): 
		if prev_answers is not None:
			coords_to_answer = prev_answers[idx]
			# Next, set the label ids predicted by the model
			prev_label_ids_example = token_type_ids_example[:,3] # shape (seq_len,)
			model_label_ids = np.zeros_like(prev_label_ids_example.cpu().numpy()) # shape (seq_len,)

			# for each token in the sequence:
			token_type_ids_example = token_type_ids[idx] # shape (seq_len, 7)
			for i in range(model_label_ids.shape[0]):
				segment_id = token_type_ids_example[:,0].tolist()[i]
				col_id = token_type_ids_example[:,1].tolist()[i] - 1
				row_id = token_type_ids_example[:,2].tolist()[i] - 1
				if row_id >= 0 and col_id >= 0 and segment_id == 1:
					model_label_ids[i] = int(coords_to_answer[(col_id, row_id)])

			# set the prev label ids of the example (shape (1, seq_len) )
			token_type_ids_example[:,3] = torch.from_numpy(model_label_ids).type(torch.long).to(device)   

		prev_answers = {}
		# get the example
		input_ids_example = input_ids[idx] # shape (seq_len,)
		attention_mask_example = attention_mask[idx] # shape (seq_len,)
		token_type_ids_example = token_type_ids[idx] # shape (seq_len, 7)
		# forward pass to obtain the logits
		outputs = model(input_ids=input_ids_example.unsqueeze(0), 
						attention_mask=attention_mask_example.unsqueeze(0), 
						token_type_ids=token_type_ids_example.unsqueeze(0))
		logits = outputs.logits
		all_logits.append(logits)

		# convert logits to probabilities (which are of shape (1, seq_len))
		dist_per_token = torch.distributions.Bernoulli(logits=logits)
		probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(dist_per_token.probs.device) 

		# Compute average probability per cell, aggregating over tokens.
		# Dictionary maps coordinates to a list of one or more probabilities
		coords_to_probs = collections.defaultdict(list)
		prev_answers = {}
		for i, p in enumerate(probabilities.squeeze().tolist()):
			segment_id = token_type_ids_example[:,0].tolist()[i]
			col = token_type_ids_example[:,1].tolist()[i] - 1
			row = token_type_ids_example[:,2].tolist()[i] - 1
			if col >= 0 and row >= 0 and segment_id == 1:
				coords_to_probs[(col, row)].append(p)

		# Next, map cell coordinates to 1 or 0 (depending on whether the mean prob of all cell tokens is > 0.5)
		coords_to_answer = {}
		for key in coords_to_probs:
			coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
		prev_answers[idx+1] = coords_to_answer
		
	logits_batch = torch.cat(tuple(all_logits), 0)
	
	return logits_batch

def linearize_table(table, question):
    output = f"Table - Columns: {' | '.join(table['header'])}"
    for i, val in enumerate(table['rows']):
        output += f" row {i+1}: {' | '.join(val)}"
    task_text = f'Q: {question}\nA:'
    return output, task_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--perturbation_type", type=str, default="yilunzhao/robut")
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    from model_loaders import load_pcw_wrapper

    output_dir = f"{args.output_dir}/{args.split_name}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    # pipeline = transformers.pipeline('text-generation', model=model_id, model_kwargs={'torch_dtype': torch.bfloat16}, device_map='auto', return_full_text=False)
    

    # model = GPT2LMHeadModel.from_pretrained(args.model_name)
    # model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_bos_token=True)

    dataset = load_dataset(args.dataset_name, split=args.split_name)
    dataset = dataset.filter(lambda x: x['perturbation_type'] == 'row')
    preds, refs = [], []
    output_data = []
    error_table_ids = set()
    
    for example in tqdm(dataset):
        table = example["table"]
        table_df = pd.DataFrame.from_records(table["rows"], columns=table["header"])

        if args.split_name in ["wikisql", "wtq"]:
            pcw_wrapper = load_pcw_wrapper(args.model_name, n_windows=len(table['rows']))
            contexts, task_text = linearize_table(table, example['question'])
            print(f"\nPrompt: {contexts}\n{task_text}")
            # input_ids = tokenizer(prompt, return_tensors='pt').to('cuda')
            # if input_ids['input_ids'].shape[1] > 1000:
            #     continue
            # gen_tokens = model.generate(**input_ids, do_sample=True, top_p=0.92, top_k=0, temperature=0.1, max_length=1024)
            # gen_tokens = model.generate(**input_ids, max_length=1024)
            # gen_text = tokenizer.decode(gen_tokens[0][input_ids['input_ids'].shape[1]:])
            # pred = gen_text.split('\n')[0].replace('<|endoftext|>','').strip()
            # pred = pipeline(prompt)[0]['generated_text'].split('\n')[0]
            pred = pcw_wrapper.pcw_generate(contexts=[contexts], task_text=task_text, 
                                            temperature=0.01, do_sample=True, max_new_tokens=20)
            output_example = example.copy()
            output_example["prediction"] = pred
            print(f"\nTrue Answer: {example['answers']} \nPred: {pred}\n")
            ref_answer = ", ".join(example["answers"])
            acc = get_denotation_accuracy([pred], [ref_answer])  # !!

            refs.append(ref_answer)
            preds.append(pred)
            output_example["accuracy"] = acc
            output_data.append(output_example)
        else:
            inputs = tokenizer(table=table_df, 
						queries=ast.literal_eval(example["question"]),
						truncation=True,
						padding="max_length",
						return_tensors="pt")
            logits = compute_prediction_sequence(model, inputs, device)
            predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits.cpu().detach())

            answers = []
            for coordinates in predicted_answer_coordinates:
                if len(coordinates) == 1:
                    # only a single cell:
                    answer = table_df.iat[coordinates[0]]
                else:
                    # multiple cells
                    cell_values = []
                    for coordinate in coordinates:
                        cell_values.append(table_df.iat[coordinate])
                    answer = ", ".join(cell_values)
                # replace \u00a0 with space
                answer = answer.replace("\u00a0", " ")
                answer = answer.replace("\"", "")
                answers.append(answer)
                preds.append(answer)
            
            ref_answers = [", ".join(ast.literal_eval(answer)) for answer in example["answers"]]
            accuracy = get_sqa_denotation_accuracy(answers, ref_answers)
            refs.extend(ref_answers)

            output_example = example.copy()
            output_example["prediction"] = answers
            output_example["accuracy"] = accuracy
            output_data.append(output_example)

    assert len(preds) == len(refs)
    output_path = f"{output_dir}/tapas-preds.json"
    json.dump(output_data, open(output_path, "w"), indent=4)

    if args.split_name in ["wikisql", "wtq"]:
        prediction_results = calculate_final_scores(output_data)
    else:
        prediction_results = calculate_sqa_final_scores(output_data)
    print(prediction_results)
    output_prediction_file = f"{output_dir}/tapas-scores.json"
    json.dump(prediction_results, open(output_prediction_file, "w"), indent=4)

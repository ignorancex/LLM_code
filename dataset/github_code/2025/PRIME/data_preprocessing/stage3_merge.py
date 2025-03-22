import os
import json
import re
from math_util import *
from IPython import embed
from tqdm import tqdm

date = "20241209"
formatted_qa_dir = f"{date}/formatted"
formatted_qa_dataset = []

original_qa_path = f"{date}/stage1_filtered.jsonl"
original_qa_dataset = []

subsets = ["amc_aime"] # TODO

rm_stat = {}

for subset in subsets: rm_stat[subset] = {}

def tell_choice(d):
    choices = ["A", "B", "C"]
    if all([f"{choice}." in d['prompt'] for choice in choices]):
        return True
    
    if all([f"{choice}:" in d['prompt'] for choice in choices]):
        return True
    
    if all([f"({choice})" in d['prompt'] for choice in choices]):
        return True

    return False


def add_rm_stat(subset, type, d):
    if type not in rm_stat[subset]:
        rm_stat[subset][type] = {"cnt": 0, "sampled_data": []}
    
    rm_stat[subset][type]["cnt"] += 1
    if len(rm_stat[subset][type]["sampled_data"]) < 10:
        rm_stat[subset][type]["sampled_data"].append(d)

def process_fn(d):
    rm = False
    subset = d['source']
    problem = d["prompt"].lower()
    is_matched, extracted_model_output = match_answer(d["solution"])

    if "figure" in problem:
        rm = True
        add_rm_stat(subset, "figure", d)
    elif "prove" in problem or "show that" in problem:
        rm = True
        add_rm_stat(subset, "prove", d)
    elif tell_choice(d):
        rm = False
        add_rm_stat(subset, "choice", d)
    elif "\\underline{" in problem:
        rm = True
        add_rm_stat(subset, "fill-in-blank", d)
    elif not is_matched:
        rm = True
        add_rm_stat(subset, "ans_not_matched", d)

    nd = {"id": d["id"], "source": d["source"], "prompt": d["prompt"], "solution": d["solution"], "reference": extracted_model_output, "remove": rm, "type": d["type"]}
    return nd

for file in os.listdir(formatted_qa_dir):
    file_path = os.path.join(formatted_qa_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            formatted_qa_dataset.append(data)

print(f"Total number in formatted_qa_dataset: {len(formatted_qa_dataset)}")
new_dataset = []
for data in formatted_qa_dataset:
    new_dataset.append(process_fn(data))

with open(original_qa_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        original_qa_dataset.append(data)

original_qa_dataset = [data for data in original_qa_dataset if not data['remove'] and data['type'] != "choice"]
print(f"Total number in original_qa_dataset: {len(original_qa_dataset)}")

new_dataset.extend(original_qa_dataset)
new_dataset = [data for data in new_dataset if not data['remove']]
new_dataset = sorted(new_dataset, key=lambda x: int(x['id']))
print(f"Total number in new_dataset: {len(new_dataset)}")

with open(f"{date}/stage3_stat.json", 'w', encoding='utf-8') as f:
    json.dump(rm_stat, f, ensure_ascii=False, indent=4)

with open(f"{date}/stage3_final_math_qa.jsonl", 'w', encoding='utf-8') as f:
    for data in new_dataset:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

import os
import sys
import json
import random
from tqdm import tqdm
from collections import defaultdict

# Set random seed
random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from utils.prompt import answer_question_dk

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load data from {path}. Error: {e}")
        return None

def generate_dpo_data(eval_models, target_model, datasets):
    dpo_results = []
    for dataset in tqdm(datasets, desc="Processing datasets"):
        original_responses_path = {model: os.path.join(PROJECT_ROOT, f'evaluation/{dataset}/{dataset}_{model}_original.json') for model in eval_models}
        original_responses_data = {model: load_json(path) for model, path in original_responses_path.items()}
        enhanced_responses_path = {model: os.path.join(PROJECT_ROOT, f'evaluation/{dataset}/{target_model}/{model}/{dataset}_{model}_enhanced.json') for model in eval_models}
        enhanced_responses_data = {model: load_json(path) for model, path in enhanced_responses_path.items()}
        
        total_items = len(original_responses_data[eval_models[0]])
        for i in tqdm(range(total_items), desc=f"Processing {dataset} items", leave=False):
            unique_id = i
            if original_responses_data[eval_models[0]][i]['good_questions'] == []:
                continue
            for good_question in original_responses_data[eval_models[0]][i]['good_questions']:
                question = good_question['question']
                choices = good_question['choices']
                ground_truth = good_question.get('ground_truth', None) or good_question.get('answer', None)
                chosen, rejected = None, None
                chosen_model, rejected_model = None, None
                
                for eval_model in eval_models:
                    match_item = next((item for item in enhanced_responses_data[eval_model] if item['question'] == question), None)
                    if match_item is not None and match_item.get('result') is not None and match_item.get('result') == False:
                        rejected = match_item['original_response']
                        rejected_model = eval_model
                        break
                for eval_model in eval_models:
                    match_item = next((item for item in enhanced_responses_data[eval_model] if item['question'] == question), None)
                    if match_item is not None and match_item.get('result') is not None and match_item.get('result') == True:
                        chosen = match_item['original_response']
                        chosen_model = eval_model
                        break
                if chosen is not None and rejected is not None:
                    result = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": answer_question_dk(question=question, choices=choices)
                            }
                        ],
                        "chosen": {
                            'from': 'gpt',
                            "value": chosen,
                            "model": chosen_model
                        },
                        "rejected": {
                            'from': 'gpt',
                            "value": rejected,
                            "model": rejected_model
                        },
                        "question": question,
                        "choices": choices,
                        "ground_truth": ground_truth,
                        "source": dataset,
                        "unique_id": unique_id
                    }
                    dpo_results.append(result)
    return dpo_results
                 
            
def train_test_split(data, test_size=0.2):
    random.seed(42)  # Ensure consistent splitting
    grouped_data = defaultdict(list)
    for item in data:
        key = (item['source'], item['unique_id'])
        grouped_data[key].append(item)
    groups = list(grouped_data.values())
    random.shuffle(groups)
    split_index = int(len(groups) * test_size)
    test_groups = groups[:split_index]
    train_groups = groups[split_index:]
    train_data = [item for group in train_groups for item in group]
    test_data = [item for group in test_groups for item in group]
    return train_data, test_data
    

if __name__ == '__main__':
    datasets = [
        'CommonsenseQA',
        'OpenbookQA',
        'TruthfulQA',
        "MMLU"
    ]
    eval_models = [
        'gpt-4o',
        'gemma-2-27B',
        'llama-3.1-8B',
        'qwen-2.5-72B',
        'claude-3.5-sonnet',
        'o1-mini'
    ]
    target_model = 'gpt-4o-mini'
    dpo_data = generate_dpo_data(eval_models, target_model, datasets)
    train_data, test_data = train_test_split(dpo_data)
    with open(os.path.join(PROJECT_ROOT, 'train/dpo_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(PROJECT_ROOT, 'train/dpo_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

import os
import json
import random
from datasets import load_dataset

dataset_mapping = {
    'MMLU-Pro': 'TIGER-Lab/MMLU-Pro',
    "CommonsenseQA": "tau/commonsense_qa",
    "TruthfulQA": "truthfulqa/truthful_qa",
    "MMLU": 'cais/mmlu'
}

def check_question(question):
    if '?' in question or '_' in question:
        if 'which' not in question.lower():
            return True
    return False

def dataset_to_json(dataset_name=None):
    if dataset_name in ['MMLU-Pro']:
        ds = load_dataset(dataset_mapping[dataset_name])
        raw_data = ds['test']
    elif dataset_name in ['CommonsenseQA']:
        ds = load_dataset(dataset_mapping[dataset_name])
        raw_data = ds['validation']
    elif dataset_name in ['TruthfulQA']:
        ds = load_dataset(dataset_mapping[dataset_name], 'multiple_choice')
        raw_data = ds['validation']
    elif dataset_name in ['MMLU']:
        ds = load_dataset(dataset_mapping[dataset_name], 'all')
        raw_data = ds['test']
    
    if dataset_name == 'MMLU':
        filtered_data = [
            item for item in raw_data
        ]
    elif dataset_name == 'OpenbookQA':
        filtered_data = [item for item in raw_data if check_question(item['question_stem'])]
    else:
        filtered_data = [item for item in raw_data if check_question(item['question'])]
    
    if len(filtered_data) < 300:
        print(f"Warning: Only {len(filtered_data)} samples match the criteria, selecting all available.")
        sampled_data = filtered_data
    else:
        random.seed(42)
        sampled_data = random.sample(filtered_data, 300)
    
    data = []
    for item in sampled_data:
        if dataset_name == 'MMLU-Pro':
            question = item['question']
            choices = item['options']
            answer = choices[item['answer_index']]
            source = 'MMLU-Pro'
            subject = item['src']
            data.append(
                {
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'source': source,
                    'subject': subject
                }
            )
            continue
        elif dataset_name == 'CommonsenseQA':
            question = item['question']
            choices = item['choices']['text']
            answer = choices[ord(item['answerKey']) - ord('A')]
            source = 'CommonsenseQA'
        elif dataset_name == 'TruthfulQA':
            question = item['question']
            choices = item['mc1_targets']['choices']
            labels = item['mc1_targets']['labels']
            answer = choices[labels.index(1)]
            source = 'TruthfulQA'
        elif dataset_name == 'MMLU':
            question = item['question']
            choices = item['choices']
            answer = choices[item['answer']]
            source = 'MMLU'
            subject = item['subject']
            data.append({
                'question': question,
                'choices': choices,
                'answer': answer,
                'source': source,
                'subject': subject
            })
            continue
        data.append({
            'question': question,
            'choices': choices,
            'answer': answer,
            'source': source
        })

    output_file = f"data/{dataset_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Filtered data has been saved to {output_file}")

if __name__ == '__main__':
    dataset_to_json('MMLU')
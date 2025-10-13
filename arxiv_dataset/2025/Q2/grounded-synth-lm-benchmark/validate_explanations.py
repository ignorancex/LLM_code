import os
import json
import numpy as np
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts
import answer_parser
from model_interface import SglModelAsync


def build_choices_string(choices):
    return '\n'.join([f"{choice}: {choices[choice]}" for i, choice in enumerate(choices)])


def compute_scores(dataset_fp, open_ended, remote, model, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'answer_correctness_score' not in d or d['answer_correctness_score'] is None:
            any_missing = True
        if 'explanation_validity_score' not in d or d['explanation_validity_score'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    print(f"Computing explanation validity score for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts  EXPLANATION_VALIDATION_OPEN_PROMPT
    if open_ended:
        model_prompts = [prompts.EXPLANATION_VALIDATION_OPEN_PROMPT.format(context=d.get('context', ''), question=d['question'], answer=d['answer'], explanation=d['explanation']) for d in dataset]
    else:
        model_prompts = [prompts.EXPLANATION_VALIDATION_PROMPT.format(context=d.get('context', ''), question=d['question'], choices=build_choices_string(d['choices']), answer=d['answer'], explanation=d['explanation']) for d in dataset]

    model = SglModelAsync(remote=remote, model=model)
    results, total_time = model.generate(model_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")

    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            parsed = answer_parser.parse_explanation_validity_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
                dataset[i]['answer_correctness_score'] = None  
                dataset[i]['explanation_validity_score'] = None
            else:
                dataset[i]['answer_correctness_score'] = parsed['answer_correctness_score']
                dataset[i]['explanation_validity_score'] = parsed['explanation_validity_score']


    scores = [d['answer_correctness_score'] for d in dataset]
    print(f"Average answer correctness score: {np.mean(scores)}")

    scores = [d['explanation_validity_score'] for d in dataset]
    print(f"Average explanation validity score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)



def validate_explanations(ifp, open_ended=False,remote='opchat', model='Llama-4-Maverick-17B-128E-Instruct-FP8'):

    fns = []
    for fn in os.listdir(ifp):
        if os.path.isdir(os.path.join(ifp, fn)) and ('novel' in fn or 'reformat' in fn):
            # Look for json files within these directories
            dataset_dir = os.path.join(ifp, fn)
            for file in os.listdir(dataset_dir):
                if file.endswith('.jsonl'):
                    # fns.append(os.path.join(dataset_dir, file))
                    fns.append(file.replace('.jsonl', ''))
    fns.sort()
    if len(fns) == 0:
        print(f"validate_explanations: No datasets found in {ifp}")
        return
    print(f"validate_explanations: Found {len(fns)} datasets")
    print(f"validate_explanations: Datasets: {fns}")



    force_flag = False
    print(f"validate_explanations: Validating answer correctness and Explanation validity for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")
        compute_scores(dataset_fp, open_ended,remote, model, force=force_flag)
    



    avg_answer_correctness_score = {}
    avg_explanation_validity_score = {}
    for fn in fns:
        dataset_fp = f"{ifp}/{fn}/{fn}.jsonl"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)

        invalid_idx = list()
        for i in range(len(data)):
            d = data[i]
            if d.get('answer_correctness_score') is None or not (1 <= d['answer_correctness_score'] <= 10):
                invalid_idx.append(i)
            if d.get('explanation_validity_score') is None or not (1 <= d['explanation_validity_score'] <= 10):
                invalid_idx.append(i)

        
        # Print invalid data entries
        if invalid_idx:
            print(f"Found {len(invalid_idx)} invalid entries in dataset {fn}:")
            for idx in invalid_idx: 
                print(f"  Entry {idx}:")
                print(f"    Question: {data[idx]['question']}")
                print(f"    Answer Correctness: {data[idx].get('answer_correctness_score', 'missing')}")
                print(f"    Explanation Validity: {data[idx].get('explanation_validity_score', 'missing')}")
                print()

            raise Exception(f"Found {len(invalid_idx)} invalid entries in dataset {fn}")
        
        
        avg_answer_correctness_score[fn] = np.mean([d['answer_correctness_score'] for d in data])
        avg_explanation_validity_score[fn] = np.mean([d['explanation_validity_score'] for d in data])

        print(f"Dataset: {os.path.basename(fn)}")
        print(f"  Average answer correctness score: {avg_answer_correctness_score[fn]:.4f}")
        print(f"  Average explanation validity score: {avg_explanation_validity_score[fn]:.4f}")
        print()



if __name__ == '__main__':
    ifp = './data-subset-1k-v0-L3-70B'
    validate_explanations(ifp)








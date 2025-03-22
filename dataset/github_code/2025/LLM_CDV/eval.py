import os
import json
import argparse
from utils.tools import get_response, clear_json
from utils.prompt import answer_question_dk, extract_answer_dk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def generate_test_cases(mode='original', data_path=None):
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load data from {data_path}. Error: {e}")
        return None
    
    new_data = []
    for item in data:
        if mode == 'original':
            new_data.append(item)
        elif mode == 'enhanced':
            if item['good_questions'] != []:
                for good_question in item['good_questions'][:1]:
                    new_data.append(good_question)
    return new_data

def single_test(model='gpt-4o', item=None):
    if item.get('result', None) is not None:
        return item
    question = item['question']
    answer = item.get('answer', None) or item.get('ground_truth', None)
    choices = item['choices']
    answer_prompt = answer_question_dk(question=question, choices=choices)
    original_response = get_response(model=model, prompt=answer_prompt, temperature=0.0001)
    extracted_prompt = extract_answer_dk(question=question, answer=original_response, choices=choices)

    try:
        extracted_response = clear_json(get_response(model='gpt-4o-mini', prompt=extracted_prompt, temperature=0.0001))
        extracted_response = json.loads(extracted_response)['final_answer']
        if extracted_response in choices and extracted_response == answer:
            item['original_response'] = original_response
            item['extracted_response'] = extracted_response
            item['result'] = True
            return item
        else:
            item['original_response'] = original_response
            item['extracted_response'] = extracted_response
            item['result'] = False
            return item
    except json.JSONDecodeError:
        return None

def main(mode='original', data_path=None, model='gpt-4o', target_model='gpt-4o-mini', max_workers=15):
    base_name = os.path.basename(data_path)
    dataset_name = base_name.replace(f'_enhanced_{target_model}.json', '')
    
    if mode == 'original':
        output_path = os.path.join('evaluation', f'{dataset_name}', f'{dataset_name}_{model}_{mode}.json')
    else:
        output_path = os.path.join('evaluation', f'{dataset_name}', f'{target_model}', f'{model}', f'{dataset_name}_{model}_{mode}.json')

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping...")
        return
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_cases = generate_test_cases(mode=mode, data_path=data_path)
    if not test_cases:
        print("No test cases generated.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(single_test, model, item): item for item in test_cases}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing test cases"):
            result = future.result()
            if result:
                results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Test cases saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation with specified parameters')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--target_model', type=str, required=True, help='Target model name')
    parser.add_argument('--eval_models', nargs='+', required=True, help='List of evaluation models')
    parser.add_argument('--modes', nargs='+', choices=['original', 'enhanced'], required=True, help='Modes to process')
    
    args = parser.parse_args()
    
    # Generate dataset paths
    datasets_path = [f'enhanced/{args.target_model}/{dataset}_enhanced_{args.target_model}.json' for dataset in args.datasets]
    
    # Process each combination
    for dataset_path in datasets_path:
        for model in args.eval_models:
            for mode in args.modes:
                print(f"Processing \033[92m{dataset_path}\033[0m with \033[94m{model}\033[0m in \033[93m{mode}\033[0m mode.")
                main(
                    mode=mode,
                    data_path=dataset_path,
                    model=model,
                    target_model=args.target_model
                )
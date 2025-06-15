from openai import OpenAI
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
models = ['deepseek-reasoner']
languages = ['cpp']
input_path = 'LLM_code/codeforces/subset_select/benchmark.jsonl'
client = OpenAI(base_url='https://api.deepseek.com/beta', api_key='')

def generate_code(prompt, idx, model, lang):
    try:
        prefix = f'\n```{lang}\n'
        completion = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': prefix, 'prefix': True}], stop=['```'])
        code = completion.choices[0].message.content.strip()
    except Exception as e:
        code = f'[Error: {e}]'
    return (idx, code)
with open(input_path, 'r', encoding='utf-8') as fin:
    problems = [json.loads(line) for line in fin if line.strip()]
for model in models:
    for lang in languages:
        output_path = f"LLM_code/codeforces/subset_select/deepseek_{model.split('-')[-1]}_{lang}.jsonl"
        existing_results = {}
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        existing_results[obj['problem']] = obj
        with open(output_path, 'w', encoding='utf-8') as fout:
            for item in tqdm(problems, desc=f'Processing {model} / {lang}'):
                problem = item.get('problem', '')
                context = item.get('context_plain', '').strip()
                prompt = f'Your task is to carefully read the following problem description and implement a solution in {lang}. Return only the code without any explanations. Here is the problem description:\n\n{context}'
                if problem in existing_results:
                    result = existing_results[problem]
                    missing_or_error_indices = [i for i in range(1, 33) if f'pass@{i}' not in result or str(result[f'pass@{i}']).startswith('[Error')]
                    if not missing_or_error_indices:
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        continue
                else:
                    result = {'problem': problem}
                    missing_or_error_indices = list(range(1, 33))
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {executor.submit(generate_code, prompt, i, model, lang): i for i in missing_or_error_indices}
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Fixing {problem}', leave=False, ncols=80):
                        (idx, code) = future.result()
                        result[f'pass@{idx}'] = code
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()
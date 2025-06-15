from openai import OpenAI
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key='')
input_path = 'LLM_code/codeforces/subset_select/benchmark.jsonl'
output_path = 'LLM_code/codeforces/subset_select/qwen_14b_cpp.jsonl'
existing_results = {}
try:
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                existing_results[obj['problem']] = obj
except FileNotFoundError:
    pass

def generate_code(prompt, idx):
    try:
        completion = client.chat.completions.create(extra_headers={'HTTP-Referer': '<YOUR_SITE_URL>', 'X-Title': '<YOUR_SITE_NAME>'}, extra_body={'enable_thinking': False}, model='qwen3-14b', messages=[{'role': 'user', 'content': prompt}])
        code = completion.choices[0].message.content.strip()
    except Exception as e:
        code = f'[Error: {e}]'
    return (idx, code)
with open(input_path, 'r', encoding='utf-8') as fin:
    problems = [json.loads(line) for line in fin if line.strip()]
with open(output_path, 'w', encoding='utf-8') as fout:
    for item in tqdm(problems, desc='Processing problems'):
        problem = item.get('problem', '')
        context = item.get('context_plain', '').strip()
        prompt = f'Your task is to carefully read the following problem description and implement a solution in C++. Return only the code without any explanations. Here is the problem description:\n\n{context}'
        if problem in existing_results:
            result = existing_results[problem]
            missing_or_error_indices = [i for i in range(1, 33) if f'pass@{i}' not in result or str(result[f'pass@{i}']).startswith('[Error')]
            if not missing_or_error_indices:
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                continue
        else:
            result = {'problem': problem}
            missing_or_error_indices = list(range(1, 33))
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(generate_code, prompt, i): i for i in missing_or_error_indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Fixing {problem}', leave=False, ncols=80):
                (idx, code) = future.result()
                result[f'pass@{idx}'] = code
        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
        fout.flush()
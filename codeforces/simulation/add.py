import json
import random
from pathlib import Path
extract_path = Path('LLM_code/codeforces/simulation/deepseek_32b_python_extract.json')
models_path = Path('LLM_code/codeforces/models_code/deepseek_reasoner_py.jsonl')
with open(extract_path, 'r', encoding='utf-8') as f:
    extract_data = json.load(f)
models_code = []
with open(models_path, 'r', encoding='utf-8') as f:
    for line in f:
        models_code.append(json.loads(line))
problem_pass_map = {}
for item in models_code:
    problem = item.get('problem', '').strip()
    if problem:
        passes = [v for (k, v) in item.items() if k.startswith('pass@') and isinstance(v, str)]
        if passes:
            problem_pass_map.setdefault(problem, []).extend(passes)
original_empty = 0
filled_count = 0
for item in extract_data:
    code_block = item.get('generate_code_block', '').strip()
    if not code_block:
        original_empty += 1
        fullname = item.get('fullname', '').strip()
        candidates = problem_pass_map.get(fullname, [])
        if candidates:
            item['generate_code_block'] = random.choice(candidates)
            filled_count += 1
output_path = extract_path.with_name('deepseek_32b_python_extract_filled.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(extract_data, f, indent=2, ensure_ascii=False)
import json
import ast
from pathlib import Path
input_path = Path('LLM_code/codeforces/simulation/valid/gemma_python.json')
output_path = Path('LLM_code/codeforces/simulation/valid/gemma_python_1.json')
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
valid_data = []
for item in data:
    code_block = item.get('generate_code_block', '').strip()
    ref_block = item.get('generate_ref_code_block', '').strip()
    if code_block and ref_block:
        if is_valid_python(code_block) and is_valid_python(ref_block):
            valid_data.append(item)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)
print(len(valid_data))
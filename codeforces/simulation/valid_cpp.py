import json
from pathlib import Path
input_path = Path('LLM_code/codeforces/simulation/gemma_27b_cpp_extract.json')
output_path = Path('LLM_code/codeforces/simulation/valid/gemma_27b_cpp_valid.json')
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
valid_data = []
for item in data:
    code_block = item.get('generate_code_block', '').strip()
    ref_block = item.get('generate_ref_code_block', '').strip()
    if code_block and ref_block:
        valid_data.append(item)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)
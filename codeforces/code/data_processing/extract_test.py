import json
with open('simulation/deepseek_32b_cpp_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
empty_count = 0
for item in data:
    code = item.get('generate_code', '').strip()
    code_block = item.get('generate_code_block', '').strip()
    ref_code = item.get('generate_code_ref', '').strip()
    ref_code_block = item.get('generate_ref_code_block', '').strip()
    if not code_block and code:
        empty_count += 1
    if not ref_code_block and ref_code:
        empty_count += 1
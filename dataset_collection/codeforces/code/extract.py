import json
import re

def extract_reasoning_and_code(text):
    reasoning_match = re.search(r'### Reasoning\s*(.*?)\s*### Code', text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    code_match = re.search(r'### Code\s*```[a-zA-Z]*\s*(.*?)```', text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ''
    return reasoning, code

with open('LLM_code/codeforces/simulation/models/Qwen_cpp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    # 检查是否所有字段都已存在
    if all(k in item for k in [
        'generate_reasoning', 'generate_code_block',
        'generate_ref_reasoning', 'generate_ref_code_block'
    ]):
        continue  # 已处理，跳过

    gen = item.get('generate_code', '')
    ref = item.get('generate_code_ref', '')

    reasoning, code = extract_reasoning_and_code(gen)
    ref_reasoning, ref_code = extract_reasoning_and_code(ref)

    item['generate_reasoning'] = reasoning
    item['generate_code_block'] = code
    item['generate_ref_reasoning'] = ref_reasoning
    item['generate_ref_code_block'] = ref_code

with open('LLM_code/codeforces/simulation/temp/Qwen_cpp.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

import json
import re

def extract_reasoning_and_code(text):
    reasoning_match = re.search('### Reasoning\\s*(.*?)\\s*### Code', text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    code_match = re.search('### Code\\s*```[a-zA-Z]*\\s*(.*?)```', text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ''
    return (reasoning, code)
with open('simulation/deepseek_32b_cpp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    gen = item.get('generate_code', '')
    ref = item.get('generate_code_ref', '')
    (reasoning, code) = extract_reasoning_and_code(gen)
    (ref_reasoning, ref_code) = extract_reasoning_and_code(ref)
    item['generate_reasoning'] = reasoning
    item['generate_code_block'] = code
    item['generate_ref_reasoning'] = ref_reasoning
    item['generate_ref_code_block'] = ref_code
with open('simulation/deepseek_32b_cpp_1.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
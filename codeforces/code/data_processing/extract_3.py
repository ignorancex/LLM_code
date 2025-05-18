import json
import re

def extract_cpp_block(text):
    cpp_match = re.search('```cpp\\s*(.*?)\\s*```', text, re.DOTALL)
    if cpp_match:
        return cpp_match.group(1).strip()
    return ''
with open('simulation/deepseek_32b_cpp_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    if not item.get('generate_code_block', '').strip():
        gen = item.get('generate_code', '')
        cpp_code = extract_cpp_block(gen)
        if cpp_code:
            item['generate_code_block'] = cpp_code
    if not item.get('generate_ref_code_block', '').strip():
        ref = item.get('generate_code_ref', '')
        ref_cpp_code = extract_cpp_block(ref)
        if ref_cpp_code:
            item['generate_ref_code_block'] = ref_cpp_code
with open('simulation/deepseek_32b_cpp_extract.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
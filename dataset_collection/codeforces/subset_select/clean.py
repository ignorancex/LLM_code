import json
import re
from pathlib import Path
from tqdm import tqdm
input_path = 'LLM_code/codeforces/subset_select/qwen_coder_cpp.jsonl'
output_path = 'LLM_code/codeforces/models_code/qwen_coder_cpp.jsonl'

def detect_language_by_filename(filename):
    if '_py' in filename:
        return 'python'
    elif '_cpp' in filename:
        return 'cpp'
    else:
        return ''

def extract_code_block(text, language):
    if not isinstance(text, str):
        return '[Error: no code]'
    pattern = f'```{language}\\s*\\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else '[Error: no code]'

def clean_jsonl_file(input_path, output_path):
    language = detect_language_by_filename(input_path)
    cleaned_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Cleaning'):
            if not line.strip():
                continue
            obj = json.loads(line)
            problem = obj.get('problem', '')
            cleaned_obj = {'problem': problem}
            pass_entries = [(int(k.split('@')[1]), k) for k in obj if k.startswith('pass@')]
            pass_entries.sort()
            for (idx, key) in pass_entries:
                content = obj.get(key, '')
                code = extract_code_block(content, language)
                cleaned_obj[f'pass@{idx}'] = code
            cleaned_data.append(cleaned_obj)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in cleaned_data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    clean_jsonl_file(input_path, output_path)
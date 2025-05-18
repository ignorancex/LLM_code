import json
input_file = 'LLM_code/codeforces/cf_code.json'
output_file = 'LLM_code/codeforces/cf_code_clean.json'
sensitive_token = '353388068:AAE-N_3Ic7rD8EMTv-wgofoBscJT_ofwbG4'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f, strict=False)
filtered_data = [item for item in data if sensitive_token not in item.get('sourceCode', '')]
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
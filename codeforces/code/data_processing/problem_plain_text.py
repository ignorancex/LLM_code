import json
from bs4 import BeautifulSoup
from tqdm import tqdm
input_path = 'LLM_code/codeforces/cf_code_clean.json'
output_path = 'LLM_code/codeforces/cf_code_plain.json'
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in tqdm(data, desc='Extracting text from HTML'):
    html = item.get('context', '')
    soup = BeautifulSoup(html, 'html.parser')
    text = '\n'.join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    item['context_plain'] = text
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
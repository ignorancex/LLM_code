import urllib.request
from bs4 import BeautifulSoup
import json
import os

def get_tags(fullname):
    problemSet = ''.join(filter(str.isdigit, fullname))
    problemId = ''.join(filter(str.isalpha, fullname))
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        url = f'https://codeforces.com/problemset/problem/{problemSet}/{problemId}'
        req = urllib.request.Request(url=url, headers=headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        tags = [span.get_text(strip=True) for span in soup.find_all('span', class_='tag-box')]
        titles = [span['title'] for span in soup.find_all('span', class_='tag-box')]
        return (tags, titles)
    except Exception as e:
        return (None, None)

def load_finished(filepath):
    finished = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    finished.add(data['problem'])
                except json.JSONDecodeError:
                    continue
    return finished

def process(input_json, output_jsonl):
    with open(input_json, 'r', encoding='utf-8') as f:
        problems_data = json.load(f)
    finished = load_finished(output_jsonl)
    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
        for item in problems_data:
            fullname = item.get('fullname')
            if not fullname:
                continue
            if fullname in finished:
                continue
            (tags, titles) = get_tags(fullname)
            if tags is None:
                continue
            if len(tags) == 0:
                continue
            result = {'problem': fullname, 'tags': tags, 'tags_titles': titles}
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_f.flush()
if __name__ == '__main__':
    input_json = 'gemma_27b_python.json'
    output_jsonl = 'tags_py.jsonl'
    process(input_json, output_jsonl)
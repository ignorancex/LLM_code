import json
import requests
from collections import defaultdict

def is_link_accessible(url):
    """检查链接是否可访问"""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def load_existing_links(prefix):
    """加载已有的链接（来自旧文件）"""
    try:
        with open(f'LLM_code/dataset/github_links/link_20{prefix}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('github_links', []))
    except FileNotFoundError:
        return set()

def extract_github_links(input_file):
    """提取 GitHub 链接并写入新文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefixes = [f'2{x:01d}' for x in range(5)]
    links_by_prefix = defaultdict(list)
    for prefix in prefixes:
        count = 0
        existing_links = load_existing_links(prefix)
        for y in range(1, 13):
            sub_prefix = f'{prefix}{y:02d}'
            y_count = 0
            for item in data:
                item_id = item.get('id', '')
                github_link = item.get('github_links', '').rstrip('\\')
                if item_id.startswith(sub_prefix) and github_link:
                    if github_link in existing_links:
                        continue
                    if y_count < 50 and count < 600:
                        if is_link_accessible(github_link):
                            links_by_prefix[prefix].append(github_link)
                            y_count += 1
                            count += 1
                        if y_count >= 50 or count >= 600:
                            break
                if count >= 600:
                    break
            if count >= 600:
                break
    for (prefix, links) in links_by_prefix.items():
        output_data = {'github_links': links}
        output_file = f'link_20{prefix}_new.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
extract_github_links('LLM_code/dataset/github_links/filtered_github_links.json')
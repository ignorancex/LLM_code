import json
from datetime import datetime
from collections import defaultdict
input_path = 'LLM_code/dataset/github_links/valid_links.json'
output_path = 'LLM_code/dataset/github_links/valid_links_by_quarter.json'

def get_quarter(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        quarter = (dt.month - 1) // 3 + 1
        return f'{dt.year}Q{quarter}'
    except Exception as e:
        return None
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
grouped_links = defaultdict(list)
for item in data:
    update_date = item.get('update_date')
    github_link = item.get('github_links')
    quarter = get_quarter(update_date)
    if quarter and github_link:
        grouped_links[quarter].append(github_link)

def sort_quarters(quarter_keys):

    def quarter_sort_key(q):
        (year, qtr) = q.split('Q')
        return int(year) * 10 + int(qtr)
    return sorted(quarter_keys, key=quarter_sort_key)
sorted_grouped_links = {quarter: grouped_links[quarter] for quarter in sort_quarters(grouped_links.keys())}
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_grouped_links, f, indent=4)
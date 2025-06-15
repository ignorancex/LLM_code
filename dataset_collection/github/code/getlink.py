import json
from collections import defaultdict

def extract_github_links(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefixes = [f'2{x:01d}' for x in range(5)]
    links_by_prefix = defaultdict(list)
    for prefix in prefixes:
        count = 0
        for y in range(1, 13):
            sub_prefix = f'{prefix}{y:02d}'
            y_count = 0
            for item in data:
                item_id = item.get('id', '')
                if item_id.startswith(sub_prefix) and 'github_links' in item:
                    cleaned_links = [link.rstrip('\\') for link in item['github_links']]
                    if cleaned_links:
                        first_link = cleaned_links[0]
                        if y_count < 100 and count < 1200:
                            links_by_prefix[prefix].append(first_link)
                            y_count += 1
                            count += 1
                        if y_count >= 100 or count >= 1200:
                            break
                if count >= 1200:
                    break
            if count >= 1200:
                break
    for (prefix, links) in links_by_prefix.items():
        output_data = {'github_links': links}
        output_file = f'link_20{prefix}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
extract_github_links('github_links.json')
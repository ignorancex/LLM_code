import json
with open('LLM_code/code/github_links/filtered_github_links.json', 'r') as f:
    metadata = json.load(f)
link_to_category = {}
for item in metadata:
    link = item.get('github_links', '').strip()
    if link:
        categories = item.get('categories', '').strip()
        if categories:
            first_category = categories.split()[0]
            link_to_category[link] = first_category
with open('LLM_code/code/github_links/python_dataset_links_new.json', 'r') as f:
    quarter_data = json.load(f)
result = {}
for (quarter, links) in quarter_data.items():
    result[quarter] = []
    for link in links:
        if link in link_to_category:
            result[quarter].append({'link': link, 'categories': link_to_category[link]})
with open('quarter_links_with_categories.json', 'w') as f:
    json.dump(result, f, indent=4)
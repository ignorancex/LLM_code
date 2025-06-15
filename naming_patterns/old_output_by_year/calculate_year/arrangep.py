import os
import json
output_dir = 'output_package_asone'
years = [2020, 2021, 2022, 2023, 2024, 2025]
merged_data = {}
for year in years:
    folder_path = os.path.join(output_dir, str(year))
    package_file = os.path.join(folder_path, 'package.json')
    with open(package_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for (key, value) in data.items():
        if key not in merged_data:
            merged_data[key] = {'count': [], 'methods': {}}
        merged_data[key]['count'].append({year: value['count']})
        for (method, method_count) in value['methods'].items():
            if method not in merged_data[key]['methods']:
                merged_data[key]['methods'][method] = {}
            merged_data[key]['methods'][method][year] = method_count
for key in merged_data:
    merged_data[key]['count_total'] = sum([x.get(year, 0) for year in range(2020, 2026) for x in merged_data[key]['count']])
    for method in merged_data[key]['methods']:
        merged_data[key]['methods'][method]['total'] = sum(merged_data[key]['methods'][method].values())
sorted_data = dict(sorted(merged_data.items(), key=lambda item: item[1]['count_total'], reverse=True))
with open('package_byproject_asone.json', 'w', encoding='utf-8') as out_file:
    json.dump(sorted_data, out_file, ensure_ascii=False, indent=4)
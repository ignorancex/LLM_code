import os
import re
import json
from collections import defaultdict, Counter

def extract_imports_and_calls(file_path):
    imports = Counter()
    calls = defaultdict(Counter)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    import_pattern = re.compile('^(?:from\\s+([\\w.]+)\\s+import\\s+([\\w., *]+)|import\\s+([\\w.]+)(?:\\s+as\\s+([\\w]+))?)')
    call_pattern = re.compile('\\b([a-zA-Z_]\\w*)\\.([a-zA-Z_]\\w*)\\b')
    local_aliases = {}
    for line in lines:
        line = line.strip()
        match = import_pattern.match(line)
        if match:
            if match.group(3):
                module = match.group(3)
                alias = match.group(4) if match.group(4) else module.split('.')[0]
                imports[module.split('.')[0]] += 1
                local_aliases[alias] = module.split('.')[0]
            elif match.group(1) and match.group(2):
                module = match.group(1)
                imported_items = match.group(2).split(',')
                base_module = module.split('.')[0]
                imports[base_module] += 1
                for item in imported_items:
                    item = item.strip()
                    if item != '*':
                        local_aliases[item] = base_module
        for match in call_pattern.finditer(line):
            (prefix, func) = match.groups()
            if prefix in imports:
                calls[prefix][func] += 1
            elif prefix in local_aliases:
                calls[local_aliases[prefix]][func] += 1
    return (imports, calls)

def analyze_directory(root_dir):
    total_imports = Counter()
    total_calls = defaultdict(Counter)
    for (subdir, _, files) in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(subdir, file)
                (imports, calls) = extract_imports_and_calls(file_path)
                total_imports.update(imports)
                for (key, value) in calls.items():
                    total_calls[key].update(value)
    return (total_imports, total_calls)

def save_results_to_json(imports, calls, output_file):
    result = {}
    sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)
    for (package, count) in sorted_imports:
        sorted_methods = {f'{package}.{func}': count for (func, count) in sorted(calls[package].items(), key=lambda x: x[1], reverse=True)}
        result[package] = {'count': count, 'methods': sorted_methods}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    directory = './github_code/2025'
    output_json = './output_package/2025/package.json'
    (imports, calls) = analyze_directory(directory)
    save_results_to_json(imports, calls, output_json)
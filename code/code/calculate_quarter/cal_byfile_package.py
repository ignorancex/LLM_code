import os
import re
import json
from collections import defaultdict, Counter
from tqdm import tqdm

def determine_quarter(year, month):
    if year < 2020:
        return (2020, 1)
    if year > 2025:
        return (2025, 1)
    if year == 2025:
        return (2025, 1)
    if 1 <= month <= 3:
        return (year, 1)
    elif 4 <= month <= 6:
        return (year, 2)
    elif 7 <= month <= 9:
        return (year, 3)
    else:
        return (year, 4)


def extract_imports_and_calls(file_path):
    imports = Counter()
    calls = defaultdict(Counter)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    import_pattern = re.compile(r'^(?:from\s+([\w.]+)\s+import\s+([\w., *]+)|import\s+([\w.]+)(?:\s+as\s+([\w]+))?)')
    call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')

    local_aliases = {}

    for line in lines:
        line = line.strip()

        match = import_pattern.match(line)
        if match:
            if match.group(3):  # Regular import
                module = match.group(3)
                alias = match.group(4) if match.group(4) else module.split('.')[0]
                imports[module.split('.')[0]] += 1
                local_aliases[alias] = module.split('.')[0]
            elif match.group(1) and match.group(2):  # From import
                module = match.group(1)
                imported_items = match.group(2).split(',')
                base_module = module.split('.')[0]
                imports[base_module] += 1
                for item in imported_items:
                    item = item.strip()
                    if item != '*':
                        local_aliases[item] = base_module

        for match in call_pattern.finditer(line):
            prefix, func = match.groups()
            if prefix in imports:
                calls[prefix][func] += 1
            elif prefix in local_aliases:
                calls[local_aliases[prefix]][func] += 1

    return imports, calls


def analyze_directory(root_dir):
    quarter_imports = defaultdict(Counter)
    quarter_calls = defaultdict(lambda: defaultdict(Counter))

    for year_folder in tqdm(os.listdir(root_dir), desc="Top-level folders"):
        year_path = os.path.join(root_dir, year_folder)
        if not os.path.isdir(year_path):
            continue

        for quarter_folder in os.listdir(year_path):
            quarter_path = os.path.join(year_path, quarter_folder)
            if not os.path.isdir(quarter_path):
                continue

            for project in os.listdir(quarter_path):
                project_path = os.path.join(quarter_path, project)
                if not os.path.isdir(project_path):
                    continue

                time_info_path = os.path.join(project_path, 'time_info.txt')
                if not os.path.exists(time_info_path):
                    continue

                file_quarters = {}
                with open(time_info_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split(': ')
                        if len(parts) == 2:
                            file_rel_path, timestamp = parts
                            match = re.match(r'(\d{4})-(\d{2})', timestamp)
                            if match:
                                y, m = int(match.group(1)), int(match.group(2))
                                year, quarter = determine_quarter(y, m)
                                full_path = os.path.join(project_path, file_rel_path.replace('/', os.sep))
                                file_quarters[full_path] = (year, quarter)

                for subdir, _, files in os.walk(project_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(subdir, file)
                            if file_path not in file_quarters:
                                continue

                            year, quarter = file_quarters[file_path]
                            imports, calls = extract_imports_and_calls(file_path)
                            for imp, count in imports.items():
                                quarter_imports[(year, quarter)][imp] += count
                            for key, funcs in calls.items():
                                for func, count in funcs.items():
                                    quarter_calls[(year, quarter)][key][func] += count

    return quarter_imports, quarter_calls


def save_results_to_json(quarter_imports, quarter_calls, output_dir):
    for (year, quarter), imports in quarter_imports.items():
        result = {}
        sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)

        for package, count in sorted_imports:
            sorted_methods = {
                f"{package}.{func}": count for func, count in
                sorted(quarter_calls[(year, quarter)][package].items(), key=lambda x: x[1], reverse=True)
            }
            result[package] = {
                "count": count,
                "methods": sorted_methods
            }

        output_path = os.path.join(output_dir, str(year), f"Q{quarter}")
        os.makedirs(output_path, exist_ok=True)

        output_file = os.path.join(output_path, f"package_{year}Q{quarter}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    directory = "LLM_code/arxiv_dataset"
    output_dir = "LLM_code/output_by_quarter/by_file"

    quarter_imports, quarter_calls = analyze_directory(directory)
    save_results_to_json(quarter_imports, quarter_calls, output_dir)

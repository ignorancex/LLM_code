# import os
# import re
# import json
# from collections import defaultdict, Counter
#
#
# def extract_imports_and_calls(file_path):
#     imports = Counter()
#     calls = defaultdict(Counter)
#
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         lines = f.readlines()
#
#     import_pattern = re.compile(r'^(?:from\s+([\w.]+)\s+import\s+([\w., *]+)|import\s+([\w.]+))')
#     call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')
#
#     local_aliases = {}
#
#     for line in lines:
#         line = line.strip()
#
#         match = import_pattern.match(line)
#         if match:
#             if match.group(3):
#                 module = match.group(3)
#                 imports[module.split('.')[0]] += 1
#             elif match.group(1) and match.group(2):
#                 module = match.group(1)
#                 imported_items = match.group(2).split(',')
#                 base_module = module.split('.')[0]
#                 imports[base_module] += 1
#                 for item in imported_items:
#                     item = item.strip()
#                     if item != '*':
#                         local_aliases[item] = base_module
#
#         for match in call_pattern.finditer(line):
#             prefix, func = match.groups()
#             if prefix in imports:
#                 calls[prefix][func] += 1
#             elif prefix in local_aliases:
#                 calls[local_aliases[prefix]][func] += 1
#
#     return imports, calls
#
#
# def analyze_directory(root_dir):
#     yearly_imports = defaultdict(Counter)
#     yearly_calls = defaultdict(lambda: defaultdict(Counter))
#
#     for year_folder in os.listdir(root_dir):
#         year_path = os.path.join(root_dir, year_folder)
#         print(year_path)
#         if not os.path.isdir(year_path):
#             continue
#
#         for project in os.listdir(year_path):
#             project_path = os.path.join(year_path, project)
#             if not os.path.isdir(project_path):
#                 continue
#
#             time_info_path = os.path.join(project_path, 'time_info.txt')
#             if not os.path.exists(time_info_path):
#                 continue
#
#             file_years = {}
#             with open(time_info_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 for line in f:
#                     parts = line.strip().split(': ')
#                     if len(parts) == 2:
#                         file_path, timestamp = parts
#                         pt=os.path.join(project_path, file_path)
#                         pt=pt.replace('/', '\\')
#                         file_years[pt] = timestamp[:4]
#                         #print(pt)
#
#             for subdir, _, files in os.walk(project_path):
#                 for file in files:
#                     if file.endswith('.py'):
#                         file_path = os.path.join(subdir, file)
#                         file_path=file_path.replace('/', '\\')
#                         if file_path not in file_years:
#                             print("error")
#                             #print("path: "+file_path)
#                             continue
#
#                         year = file_years[file_path]
#                         imports, calls = extract_imports_and_calls(file_path)
#                         for imp, count in imports.items():
#                             yearly_imports[year][imp] += count
#                         for key, funcs in calls.items():
#                             for func, count in funcs.items():
#                                 yearly_calls[year][key][func] += count
#
#     return yearly_imports, yearly_calls
#
#
# def save_results_to_json(yearly_imports, yearly_calls, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#
#     for year, imports in yearly_imports.items():
#         result = {}
#         sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)
#
#         for package, count in sorted_imports:
#             sorted_methods = {f"{package}.{func}": count for func, count in
#                               sorted(yearly_calls[year][package].items(), key=lambda x: x[1], reverse=True)}
#             result[package] = {
#                 "count": count,
#                 "methods": sorted_methods
#             }
#
#         output_file = os.path.join(output_dir, f"package_{year}.json")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=4, ensure_ascii=False)
#
#
# if __name__ == "__main__":
#     directory = "./github_code"  # 需要分析的目录
#     output_dir = "./output_file_package"  # 输出目录
#
#     yearly_imports, yearly_calls = analyze_directory(directory)
#     save_results_to_json(yearly_imports, yearly_calls, output_dir)


import os
import re
import json
from collections import defaultdict, Counter


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
    yearly_imports = defaultdict(Counter)
    yearly_calls = defaultdict(lambda: defaultdict(Counter))

    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        print(year_path)
        if not os.path.isdir(year_path):
            continue

        for project in os.listdir(year_path):
            project_path = os.path.join(year_path, project)
            if not os.path.isdir(project_path):
                continue

            time_info_path = os.path.join(project_path, 'time_info.txt')
            if not os.path.exists(time_info_path):
                continue

            file_years = {}
            with open(time_info_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        file_path, timestamp = parts
                        pt=os.path.join(project_path, file_path)
                        pt=pt.replace('/', '\\')
                        file_years[pt] = timestamp[:4]
                        #print(pt)

            for subdir, _, files in os.walk(project_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(subdir, file)
                        file_path=file_path.replace('/', '\\')
                        if file_path not in file_years:
                            print("error")
                            #print("path: "+file_path)
                            continue

                        year = file_years[file_path]
                        imports, calls = extract_imports_and_calls(file_path)
                        for imp, count in imports.items():
                            yearly_imports[year][imp] += count
                        for key, funcs in calls.items():
                            for func, count in funcs.items():
                                yearly_calls[year][key][func] += count

    return yearly_imports, yearly_calls


def save_results_to_json(yearly_imports, yearly_calls, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for year, imports in yearly_imports.items():
        result = {}
        sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)

        for package, count in sorted_imports:
            sorted_methods = {f"{package}.{func}": count for func, count in
                              sorted(yearly_calls[year][package].items(), key=lambda x: x[1], reverse=True)}
            result[package] = {
                "count": count,
                "methods": sorted_methods
            }

        output_file = os.path.join(output_dir, f"package_{year}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    directory = "./github_code"  # 需要分析的目录
    output_dir = "./output_file_package"  # 输出目录

    yearly_imports, yearly_calls = analyze_directory(directory)
    save_results_to_json(yearly_imports, yearly_calls, output_dir)
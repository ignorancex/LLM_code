# import os
# import re
# import json
# from collections import defaultdict, Counter
#
#
# def extract_imports_and_calls(file_path):
#     imports = set()
#     calls = defaultdict(set)
#
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         lines = f.readlines()
#
#     # 解析 import 语句和函数调用
#     import_pattern = re.compile(r'^(?:from\s+([\w.]+)\s+import\s+([\w., *]+)|import\s+([\w.]+))')
#     call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')
#
#     local_aliases = {}
#
#     for line in lines:
#         line = line.strip()
#
#         # 解析 import 语句
#         match = import_pattern.match(line)
#         if match:
#             if match.group(3):  # import module
#                 module = match.group(3)
#                 imports.add(module.split('.')[0])
#             elif match.group(1) and match.group(2):  # from module import X
#                 module = match.group(1)
#                 imported_items = match.group(2).split(',')
#                 base_module = module.split('.')[0]
#                 imports.add(base_module)
#                 for item in imported_items:
#                     item = item.strip()
#                     if item != '*':
#                         local_aliases[item] = base_module  # 记录别名
#
#         # 解析方法调用
#         for match in call_pattern.finditer(line):
#             prefix, func = match.groups()
#             if prefix in imports:
#                 calls[prefix].add(func)
#             elif prefix in local_aliases:
#                 calls[local_aliases[prefix]].add(func)
#
#     return imports, calls
#
#
# def analyze_directory(root_dir):
#     total_imports = Counter()
#     total_calls = defaultdict(Counter)
#
#     for project in os.listdir(root_dir):
#         project_path = os.path.join(root_dir, project)
#         if os.path.isdir(project_path):  # 只处理项目文件夹
#             project_imports = set()
#             project_calls = defaultdict(set)
#
#             for subdir, _, files in os.walk(project_path):
#                 for file in files:
#                     if file.endswith('.py'):
#                         file_path = os.path.join(subdir, file)
#                         imports, calls = extract_imports_and_calls(file_path)
#                         project_imports.update(imports)
#                         for key, value in calls.items():
#                             project_calls[key].update(value)
#
#             # 统计每个项目中唯一的包和调用方法
#             for imp in project_imports:
#                 total_imports[imp] += 1
#             for key, value in project_calls.items():
#                 for func in value:
#                     total_calls[key][func] += 1
#
#     return total_imports, total_calls
#
#
# def save_results_to_json(imports, calls, output_file):
#     result = {}
#
#     sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)
#
#     for package, count in sorted_imports:
#         sorted_methods = {f"{package}.{func}": count for func, count in
#                           sorted(calls[package].items(), key=lambda x: x[1], reverse=True)}
#         result[package] = {
#             "count": count,
#             "methods": sorted_methods
#         }
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=4, ensure_ascii=False)
#
#
# if __name__ == "__main__":
#     directory = "./github_code/2025"  # 需要分析的目录
#     output_json = "./output_package_asone/2025/package.json"  # 输出的 JSON 文件
#
#     imports, calls = analyze_directory(directory)
#     save_results_to_json(imports, calls, output_json)



import os
import re
import json
from collections import defaultdict, Counter


def extract_imports_and_calls(file_path):
    imports = set()
    calls = defaultdict(set)

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
                imports.add(module.split('.')[0])
                local_aliases[alias] = module.split('.')[0]
            elif match.group(1) and match.group(2):  # From import
                module = match.group(1)
                imported_items = match.group(2).split(',')
                base_module = module.split('.')[0]
                imports.add(base_module)
                for item in imported_items:
                    item = item.strip()
                    if item != '*':
                        local_aliases[item] = base_module

        for match in call_pattern.finditer(line):
            prefix, func = match.groups()
            if prefix in imports:
                calls[prefix].add(func)
            elif prefix in local_aliases:
                calls[local_aliases[prefix]].add(func)

    return imports, calls


def analyze_directory(root_dir):
    total_imports = Counter()
    total_calls = defaultdict(Counter)

    for project in os.listdir(root_dir):
        project_path = os.path.join(root_dir, project)
        if os.path.isdir(project_path):  # 只处理项目文件夹
            project_imports = set()
            project_calls = defaultdict(set)

            for subdir, _, files in os.walk(project_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(subdir, file)
                        imports, calls = extract_imports_and_calls(file_path)
                        project_imports.update(imports)
                        for key, value in calls.items():
                            project_calls[key].update(value)

            # 统计每个项目中唯一的包和调用方法
            for imp in project_imports:
                total_imports[imp] += 1
            for key, value in project_calls.items():
                for func in value:
                    total_calls[key][func] += 1

    return total_imports, total_calls


def save_results_to_json(imports, calls, output_file):
    result = {}

    sorted_imports = sorted(imports.items(), key=lambda x: x[1], reverse=True)

    for package, count in sorted_imports:
        sorted_methods = {f"{package}.{func}": count for func, count in
                          sorted(calls[package].items(), key=lambda x: x[1], reverse=True)}
        result[package] = {
            "count": count,
            "methods": sorted_methods
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    directory = "./github_code/2025"  # 需要分析的目录
    output_json = "./output_package_asone/2025/package.json"  # 输出的 JSON 文件

    imports, calls = analyze_directory(directory)
    save_results_to_json(imports, calls, output_json)
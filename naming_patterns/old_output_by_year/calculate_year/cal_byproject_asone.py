import os
import ast
import csv
from collections import Counter

def extract_code_info(file_path, skipped_files_log):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if '\x00' in code:
            with open(skipped_files_log, 'a', encoding='utf-8') as log:
                log.write(f'Skipped {file_path}: Contains null bytes\n')
            return (set(), set())
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, 'a', encoding='utf-8') as log:
            log.write(f'Skipped {file_path}: {str(e)}\n')
        return (set(), set())
    function_names = set()
    variable_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            function_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    variable_names.add(target.id)
    return (function_names, variable_names)

def scan_directory(directory, output_dir):
    function_project_count = Counter()
    variable_project_count = Counter()
    skipped_files_log = os.path.join(output_dir, 'skipped_files.txt')
    if os.path.exists(skipped_files_log):
        os.remove(skipped_files_log)
    for project_name in os.listdir(directory):
        project_path = os.path.join(directory, project_name)
        if not os.path.isdir(project_path):
            continue
        project_functions = set()
        project_variables = set()
        for (root, _, files) in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    (functions, variables) = extract_code_info(file_path, skipped_files_log)
                    project_functions.update(functions)
                    project_variables.update(variables)
        function_project_count.update(project_functions)
        variable_project_count.update(project_variables)
    save_to_csv(sorted(function_project_count.items(), key=lambda x: x[1], reverse=True), os.path.join(output_dir, 'functions.csv'), ['Function Name', 'Projects Count'])
    save_to_csv(sorted(variable_project_count.items(), key=lambda x: x[1], reverse=True), os.path.join(output_dir, 'variables.csv'), ['Variable Name', 'Projects Count'])
    return (function_project_count, variable_project_count)

def save_to_csv(sorted_data, file_path, header):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (key, value) in sorted_data:
            writer.writerow([key, value])
directory_path = './github_code/2025'
output_directory = './output_2025'
os.makedirs(output_directory, exist_ok=True)
(functions, variables) = scan_directory(directory_path, output_directory)
import os
import ast
import re
from collections import Counter

def extract_code_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = ast.parse(code)
    function_names = Counter()
    variable_names = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names[node.name] += 1
        elif isinstance(node, ast.Name):
            variable_names[node.id] += 1
    comments = re.findall('#.*', code)
    return (function_names, variable_names, comments)

def scan_directory(directory):
    total_functions = Counter()
    total_variables = Counter()
    total_comments = []
    for (root, _, files) in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                (functions, variables, comments) = extract_code_info(file_path)
                total_functions.update(functions)
                total_variables.update(variables)
                total_comments.extend(comments)
    return (total_functions, total_variables, total_comments)
directory_path = './filtered_repo/'
(functions, variables, comments) = scan_directory(directory_path)
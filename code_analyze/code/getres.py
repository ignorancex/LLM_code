import os
import ast
import re
from collections import Counter


def extract_code_info(file_path):
    """解析 Python 代码，提取函数名、变量名、注释"""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 解析 AST
    tree = ast.parse(code)

    function_names = Counter()
    variable_names = Counter()

    # 提取函数和变量名
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  # 函数定义
            function_names[node.name] += 1
        elif isinstance(node, ast.Name):  # 变量名
            variable_names[node.id] += 1

    # 提取注释
    comments = re.findall(r"#.*", code)

    return function_names, variable_names, comments


def scan_directory(directory):
    """扫描目录下所有 .py 文件，并统计函数名、变量名和注释"""
    total_functions = Counter()
    total_variables = Counter()
    total_comments = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions, variables, comments = extract_code_info(file_path)

                total_functions.update(functions)
                total_variables.update(variables)
                total_comments.extend(comments)

    return total_functions, total_variables, total_comments


# 测试运行（替换成你的目录路径）
directory_path = "./filtered_repo/"
functions, variables, comments = scan_directory(directory_path)

# 输出结果
print("函数名统计：", functions)
print("变量名统计：", variables)
print("注释示例（前5条）：", comments[:5])

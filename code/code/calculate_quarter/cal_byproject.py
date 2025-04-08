import os
import ast
import re
import csv
from collections import Counter
from tqdm import tqdm


def extract_code_info(file_path, skipped_files_log):
    """解析 Python 代码，提取函数名、变量名、注释"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "\x00" in code:
            with open(skipped_files_log, "a", encoding="utf-8") as log:
                log.write(f"Skipped {file_path}: Contains null bytes\n")
            return Counter(), Counter(), []

        tree = ast.parse(code)  # 解析 AST

    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, "a", encoding="utf-8") as log:
            log.write(f"Skipped {file_path}: {str(e)}\n")
        return Counter(), Counter(), []

    function_names = Counter()
    variable_names = Counter()
    comments = re.findall(r"#.*", code)

    docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
    docstrings += re.findall(r"'''(.*?)'''", code, re.DOTALL)
    comments.extend(docstrings)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            function_names[node.name] += 1
            if ast.get_docstring(node):
                comments.append(ast.get_docstring(node))
        elif isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    variable_names[target.id] += 1

    return function_names, variable_names, comments


def process_comments(comments):
    """处理注释文本，分词并统计词频"""
    word_freq = Counter()
    for comment in comments:
        words = re.findall(r"\b[a-zA-Z]+\b", comment.lower())
        word_freq.update(words)
    return word_freq


def scan_directory(directory, output_dir):
    """扫描目录下所有 .py 文件，并统计函数名、变量名和注释"""
    total_functions = Counter()
    total_variables = Counter()
    total_comments = []

    skipped_files_log = os.path.join(output_dir, "skipped_files.txt")

    if os.path.exists(skipped_files_log):
        os.remove(skipped_files_log)

    # 收集所有 .py 文件路径
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    for file_path in tqdm(py_files, desc="Processing .py files", leave=False):
        functions, variables, comments = extract_code_info(file_path, skipped_files_log)
        total_functions.update(functions)
        total_variables.update(variables)
        total_comments.extend(comments)

    comment_words_freq = process_comments(total_comments)

    save_to_csv(sorted(total_functions.items(), key=lambda x: x[1], reverse=True),
                os.path.join(output_dir, "functions.csv"), ["Function Name", "Frequency"])
    save_to_csv(sorted(total_variables.items(), key=lambda x: x[1], reverse=True),
                os.path.join(output_dir, "variables.csv"), ["Variable Name", "Frequency"])
    save_to_csv(sorted(comment_words_freq.items(), key=lambda x: x[1], reverse=True),
                os.path.join(output_dir, "comments_words.csv"), ["Word", "Frequency"])

    return total_functions, total_variables, comment_words_freq


def save_to_csv(sorted_data, file_path, header):
    """将统计结果保存到 CSV 文件"""
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, value in sorted_data:
            writer.writerow([key, value])


# 主程序：遍历所有季度
years = range(2020, 2026)

print("Starting quarterly analysis...\n")

for year in tqdm(years, desc="Years"):
    max_quarter = 1 if year == 2025 else 4
    for quarter in tqdm(range(1, max_quarter + 1), desc=f"{year}", leave=False):
        quarter_name = f"Q{quarter}"
        input_dir = f"LLM_code/arxiv_dataset/{year}/{quarter_name}"
        output_dir = f"LLM_code/output_by_quarter/by_project/{year}/{quarter_name}"

        if not os.path.exists(input_dir):
            tqdm.write(f"[Skipped] {input_dir} does not exist.")
            continue

        os.makedirs(output_dir, exist_ok=True)
        tqdm.write(f"[Processing] {input_dir}")
        scan_directory(input_dir, output_dir)

print("\n✅ All processing completed.")

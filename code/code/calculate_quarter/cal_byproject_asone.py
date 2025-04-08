import os
import ast
import csv
from collections import Counter
from tqdm import tqdm

"""
统计函数名和变量名在多少个repo中出现过，而不是一共出现了多少次
"""

def extract_code_info(file_path, skipped_files_log):
    """解析 Python 代码，提取函数名、变量名"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "\x00" in code:
            with open(skipped_files_log, "a", encoding="utf-8") as log:
                log.write(f"Skipped {file_path}: Contains null bytes\n")
            return set(), set()

        tree = ast.parse(code)  # 解析 AST
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(skipped_files_log, "a", encoding="utf-8") as log:
            log.write(f"Skipped {file_path}: {str(e)}\n")
        return set(), set()

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

    return function_names, variable_names


def scan_directory(directory, output_dir):
    """扫描一个季度目录，统计函数名、变量名出现在多少个项目中"""
    function_project_count = Counter()
    variable_project_count = Counter()

    skipped_files_log = os.path.join(output_dir, "skipped_files.txt")
    if os.path.exists(skipped_files_log):
        os.remove(skipped_files_log)  # 清空旧日志

    project_list = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for project_name in tqdm(project_list, desc=f"Scanning {os.path.basename(os.path.dirname(output_dir))}/{os.path.basename(output_dir)}"):
        project_path = os.path.join(directory, project_name)

        project_functions = set()
        project_variables = set()

        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    functions, variables = extract_code_info(file_path, skipped_files_log)
                    project_functions.update(functions)
                    project_variables.update(variables)

        # 统计当前项目中的唯一函数名和变量名
        function_project_count.update(project_functions)
        variable_project_count.update(project_variables)

    # 保存结果
    save_to_csv(sorted(function_project_count.items(), key=lambda x: x[1], reverse=True),
                os.path.join(output_dir, "functions.csv"), ["Function Name", "Projects Count"])
    save_to_csv(sorted(variable_project_count.items(), key=lambda x: x[1], reverse=True),
                os.path.join(output_dir, "variables.csv"), ["Variable Name", "Projects Count"])

    return function_project_count, variable_project_count


def save_to_csv(sorted_data, file_path, header):
    """将统计结果保存到 CSV 文件"""
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, value in sorted_data:
            writer.writerow([key, value])


# === 运行主逻辑：遍历每年每季度（2020~2025, 但2025只到Q1） ===
base_dir = "LLM_code/arxiv_dataset"  # 示例目录结构：github_code/2020/Q1/
output_base = "LLM_code/output_by_quarter/by_project_as_one"

os.makedirs(output_base, exist_ok=True)

for year in range(2020, 2026):
    max_quarter = 1 if year == 2025 else 4
    for q in range(1, max_quarter + 1):
        quarter_name = f"Q{q}"
        year_str = str(year)
        quarter_path = os.path.join(base_dir, year_str, quarter_name)
        if not os.path.isdir(quarter_path):
            continue

        output_dir = os.path.join(output_base, year_str, quarter_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🔍 Processing {year_str}/{quarter_name} ...")
        scan_directory(quarter_path, output_dir)
        print(f"✅ Finished {year_str}/{quarter_name}, results saved in {output_dir}")

print("\n🎉 All processing completed.")

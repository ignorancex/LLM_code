import os
import ast
import re
import csv
from collections import defaultdict
from tqdm import tqdm  # 导入 tqdm

"""
该脚本提取并统计每年项目中的函数、变量、注释等信息,
并将结果按年份存储为CSV文件,
对于无法解析的文件，它会将错误记录到日志中
"""

def parse_time_info(file_path):
    """解析 time_info.txt，返回 {python_file_path: 年份}"""
    year_mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                file_path, timestamp = parts
                year = timestamp[:4]  # 提取年份（前四位）
                year_mapping[file_path] = year
    return year_mapping

def calculate_indentation_consistency(lines):
    indent_unit_counts = {}
    total_indented_lines = 0
    for line in lines:
        stripped_line = line.lstrip()
        if not stripped_line or stripped_line.startswith(('#', '//', '/*', '*')):
            continue
        indent = line[:len(line)-len(stripped_line)]
        if indent:
            total_indented_lines += 1
            indent = indent.replace('\t', '    ')
            indent_length = len(indent)
            if indent_length in indent_unit_counts:
                indent_unit_counts[indent_length] += 1
            else:
                indent_unit_counts[indent_length] = 1

    if total_indented_lines == 0:
        return 1.0

    most_common_indent_count = max(indent_unit_counts.values())
    consistency = most_common_indent_count / total_indented_lines
    return consistency

def calculate_avg_function_length(lines):
    function_lengths = []
    function_pattern = r'^\s*def\s+\w+\s*\(.*\):'

    function_starts = [i for i, line in enumerate(lines) if re.match(function_pattern, line)]
    for start_line in function_starts:
        length = 0
        i = start_line
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())
            start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            if i > start_line and stripped_line and (len(line) - len(line.lstrip())) <= start_indent:
                break
            length += 1
            i += 1
        function_lengths.append(length)
    return sum(function_lengths) / len(function_lengths) if function_lengths else 0.0

def calculate_avg_nesting_depth(lines):
    nesting_depths = []
    indent_levels = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        current_indent = len(line) - len(line.lstrip())
        while indent_levels and current_indent < indent_levels[-1]:
            indent_levels.pop()
        if indent_levels and current_indent == indent_levels[-1]:
            pass
        elif current_indent > (indent_levels[-1] if indent_levels else 0):
            indent_levels.append(current_indent)
        nesting_depths.append(len(indent_levels))
    return sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0.0

def calculate_comment_ratio(lines):
    comment_lines = 0
    code_lines = 0
    in_block_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith('#'):
            comment_lines += 1
        elif re.match(r'(\'\'\'|\"\"\")', stripped_line):
            comment_lines += 1
            if stripped_line.count('\'\'\'') % 2 == 1 or stripped_line.count('\"\"\"') % 2 == 1:
                in_block_comment = not in_block_comment
        elif in_block_comment:
            comment_lines += 1
        else:
            code_lines += 1
            
    total_code_lines = code_lines + comment_lines
    return comment_lines / total_code_lines if total_code_lines > 0 else 0.0

def process_file_metrics(file_path):
    """读取 Python 文件并计算四个指标"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 计算四个指标
        metrics = {
            "indentation_consistency": calculate_indentation_consistency(lines),
            "avg_function_length": calculate_avg_function_length(lines),
            "avg_nesting_depth": calculate_avg_nesting_depth(lines),
            "comment_ratio": calculate_comment_ratio(lines)
        }
        return metrics, None

    except Exception as e:
        return None, str(e)

def scan_directory(base_directory, output_file):
    """扫描所有年份文件夹，并按更新时间归类统计"""
    year_data = defaultdict(lambda: {"metrics": defaultdict(list), "skipped_files": []})

    # 使用 tqdm 显示年份和项目的进度条
    for year_folder in tqdm(os.listdir(base_directory), desc="Scanning years"):
        year_path = os.path.join(base_directory, year_folder)
        if not os.path.isdir(year_path):
            continue

        for project in tqdm(os.listdir(year_path), desc=f"Scanning projects in {year_folder}", leave=False):
            project_path = os.path.join(year_path, project)
            if not os.path.isdir(project_path):
                continue

            time_info_path = os.path.join(project_path, "time_info.txt")
            if not os.path.exists(time_info_path):
                continue  # 没有 time_info.txt，跳过该项目

            file_year_mapping = parse_time_info(time_info_path)

            # 对每个文件进行处理，显示进度条
            for file_rel_path, year in tqdm(file_year_mapping.items(), desc=f"Processing files in {project}", leave=False):
                python_file_path = os.path.join(project_path, file_rel_path)
                if not python_file_path.endswith(".py") or not os.path.exists(python_file_path):
                    continue  # 只处理 .py 文件，且文件必须存在

                metrics, error = process_file_metrics(python_file_path)
                if error:
                    year_data[year]["skipped_files"].append(f"Skipped {python_file_path}: {error}")
                    continue

                for metric, value in metrics.items():
                    year_data[year]["metrics"][metric].append(value)

    # 计算每年的平均值
    result_data = []
    for year, data in year_data.items():
        avg_metrics = {metric: sum(values) / len(values) if values else 0.0
                       for metric, values in data["metrics"].items()}
        avg_metrics["Year"] = year
        result_data.append(avg_metrics)

    # 保存最终的结果到 CSV 文件
    save_to_csv(result_data, output_file)

    # 记录跳过的文件
    skipped_files = []
    for year, skipped_files_data in year_data.items():
        skipped_files.extend(skipped_files_data["skipped_files"])

    skipped_log = os.path.join(os.path.dirname(output_file), "skipped_files.txt")
    with open(skipped_log, "w", encoding="utf-8") as log:
        for entry in skipped_files:
            log.write(entry + "\n")

    return year_data

def save_to_csv(data, file_path):
    """将结果保存到 CSV 文件"""
    if not data:
        return
    
    header = data[0].keys()
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


# 运行
base_directory = "LLM_code/dataset/github_code"
output_file = "LLM_code/average_metrics_by_year.csv"  # 最终保存的文件

os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在

scan_directory(base_directory, output_file)

print("Processing completed. Check the 'output' folder for results.")

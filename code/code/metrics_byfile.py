import os
import ast
import re
import csv
from collections import defaultdict
from tqdm import tqdm  # Import tqdm

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
    """计算缩进一致性"""
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

def calculate_function_length(lines):
    """计算函数长度"""
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

def calculate_nesting_depth(lines):
    """计算嵌套深度"""
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
    """计算注释比例"""
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

def extract_code_info(file_path):
    """解析 Python 代码，提取代码行"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.readlines()
        return code, None
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        return None, str(e)

def process_files(base_directory):
    """处理所有 Python 文件并计算每年的四个指标的平均值"""
    year_metrics = defaultdict(lambda: {"indentation_consistency": 0.0, "function_length": 0.0, 
                                        "nesting_depth": 0.0, "comment_ratio": 0.0, 
                                        "count": 0})

    total_files = 0  # 记录文件总数
    all_files = []   # 用于存储所有的文件路径，以便给 tqdm 传递

    for year_folder in os.listdir(base_directory):
        year_path = os.path.join(base_directory, year_folder)
        if not os.path.isdir(year_path):
            continue

        for project in os.listdir(year_path):
            project_path = os.path.join(year_path, project)
            if not os.path.isdir(project_path):
                continue

            time_info_path = os.path.join(project_path, "time_info.txt")
            if not os.path.exists(time_info_path):
                continue

            file_year_mapping = parse_time_info(time_info_path)

            for file_rel_path, year in file_year_mapping.items():
                python_file_path = os.path.join(project_path, file_rel_path)
                if not python_file_path.endswith(".py") or not os.path.exists(python_file_path):
                    continue
                all_files.append(python_file_path)  # 添加到待处理文件列表
                total_files += 1

    # 使用 tqdm 进行进度条显示
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for python_file_path in all_files:
            code, error = extract_code_info(python_file_path)
            if error:
                continue

            # 计算四个指标
            indentation_consistency = calculate_indentation_consistency(code)
            function_length = calculate_function_length(code)
            nesting_depth = calculate_nesting_depth(code)
            comment_ratio = calculate_comment_ratio(code)

            # 通过文件路径解析出年份
            year = python_file_path.split(os.sep)[-3]  # 假设路径格式为 base_directory/year_folder/project/file.py

            # 将结果添加到对应年份的统计数据
            year_metrics[year]["indentation_consistency"] += indentation_consistency
            year_metrics[year]["function_length"] += function_length
            year_metrics[year]["nesting_depth"] += nesting_depth
            year_metrics[year]["comment_ratio"] += comment_ratio
            year_metrics[year]["count"] += 1

            pbar.update(1)  # 更新进度条

    # 计算每年的平均值并保存到 CSV 文件
    output_file = "code_metrics_by_year.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Average Indentation Consistency", "Average Function Length", 
                         "Average Nesting Depth", "Average Comment Ratio"])

        for year, metrics in year_metrics.items():
            if metrics["count"] > 0:
                avg_indentation_consistency = metrics["indentation_consistency"] / metrics["count"]
                avg_function_length = metrics["function_length"] / metrics["count"]
                avg_nesting_depth = metrics["nesting_depth"] / metrics["count"]
                avg_comment_ratio = metrics["comment_ratio"] / metrics["count"]
                writer.writerow([year, avg_indentation_consistency, avg_function_length, avg_nesting_depth, avg_comment_ratio])

# 运行
base_directory = "./github_code"  # 输入目录

process_files(base_directory)

print("Processing completed. Check the 'code_metrics_by_year.csv' for results.")

import os
import ast
import re
import csv
from collections import Counter, defaultdict
'\n该脚本提取并统计每年项目中的函数、变量、注释等信息,\n并将结果按年份存储为CSV文件,\n对于无法解析的文件，它会将错误记录到日志中\n'

def parse_time_info(file_path):
    """解析 time_info.txt，返回 {python_file_path: 年份}"""
    year_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                (file_path, timestamp) = parts
                year = timestamp[:4]
                year_mapping[file_path] = year
    return year_mapping

def extract_code_info(file_path):
    """解析 Python 代码，提取函数名、变量名、注释"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if '\x00' in code:
            return (None, 'Contains null bytes')
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        return (None, str(e))
    function_names = Counter()
    variable_names = Counter()
    comments = re.findall('#.*', code)
    docstrings = re.findall('"""(.*?)"""', code, re.DOTALL)
    docstrings += re.findall("'''(.*?)'''", code, re.DOTALL)
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
    return ((function_names, variable_names, comments), None)

def process_comments(comments):
    """处理注释文本，分词并统计词频"""
    word_freq = Counter()
    for comment in comments:
        words = re.findall('\\b[a-zA-Z]+\\b', comment.lower())
        word_freq.update(words)
    return word_freq

def scan_directory(base_directory, output_base):
    """扫描所有年份文件夹，并按更新时间归类统计"""
    year_data = defaultdict(lambda : {'functions': Counter(), 'variables': Counter(), 'comments': []})
    skipped_files = defaultdict(list)
    for year_folder in os.listdir(base_directory):
        year_path = os.path.join(base_directory, year_folder)
        if not os.path.isdir(year_path):
            continue
        for project in os.listdir(year_path):
            project_path = os.path.join(year_path, project)
            if not os.path.isdir(project_path):
                continue
            time_info_path = os.path.join(project_path, 'time_info.txt')
            if not os.path.exists(time_info_path):
                continue
            file_year_mapping = parse_time_info(time_info_path)
            for (file_rel_path, year) in file_year_mapping.items():
                python_file_path = os.path.join(project_path, file_rel_path)
                if not python_file_path.endswith('.py') or not os.path.exists(python_file_path):
                    continue
                (result, error) = extract_code_info(python_file_path)
                if error:
                    skipped_files[year].append(f'Skipped {python_file_path}: {error}')
                    continue
                (functions, variables, comments) = result
                year_data[year]['functions'].update(functions)
                year_data[year]['variables'].update(variables)
                year_data[year]['comments'].extend(comments)
    for (year, data) in year_data.items():
        output_dir = os.path.join(output_base, year)
        os.makedirs(output_dir, exist_ok=True)
        comment_words_freq = process_comments(data['comments'])
        save_to_csv(sorted(data['functions'].items(), key=lambda x: x[1], reverse=True), os.path.join(output_dir, f'functions_{year}.csv'), ['Function Name', 'Frequency'])
        save_to_csv(sorted(data['variables'].items(), key=lambda x: x[1], reverse=True), os.path.join(output_dir, f'variables_{year}.csv'), ['Variable Name', 'Frequency'])
        save_to_csv(sorted(comment_words_freq.items(), key=lambda x: x[1], reverse=True), os.path.join(output_dir, f'comments_words_{year}.csv'), ['Word', 'Frequency'])
        skipped_log = os.path.join(output_dir, 'skipped_files.txt')
        with open(skipped_log, 'w', encoding='utf-8') as log:
            for entry in skipped_files[year]:
                log.write(entry + '\n')
    return year_data

def save_to_csv(sorted_data, file_path, header):
    """将统计结果保存到 CSV 文件"""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (key, value) in sorted_data:
            writer.writerow([key, value])
base_directory = './github_code'
output_directory = './output'
os.makedirs(output_directory, exist_ok=True)
scan_directory(base_directory, output_directory)
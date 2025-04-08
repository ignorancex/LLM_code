import os
import ast
import re
import csv
from collections import Counter, defaultdict
from tqdm import tqdm


def determine_quarter(year, month):
    if year < 2020:
        return (2020, 1)
    if year > 2025:
        return (2025, 1)
    if year == 2025:
        return (2025, 1)
    if 1 <= month <= 3:
        return (year, 1)
    elif 4 <= month <= 6:
        return (year, 2)
    elif 7 <= month <= 9:
        return (year, 3)
    else:
        return (year, 4)


def parse_time_info(file_path, project_path):
    """解析 time_info.txt，返回 {python_file_path: (year, quarter)}"""
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                rel_path, timestamp = parts
                match = re.match(r"(\d{4})-(\d{2})", timestamp)
                if not match:
                    continue
                y, m = int(match.group(1)), int(match.group(2))
                year, quarter = determine_quarter(y, m)
                full_path = os.path.join(project_path, rel_path.replace('/', os.sep))
                mapping[full_path] = (year, quarter)
    return mapping


def extract_code_info(file_path):
    """解析 Python 代码，提取函数名、变量名、注释"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        if "\x00" in code:
            return None, "Contains null bytes"
        tree = ast.parse(code)  # 解析 AST
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        return None, str(e)

    function_names = Counter()
    variable_names = Counter()
    comments = re.findall(r"#.*", code)

    docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
    docstrings += re.findall(r"'''(.*?)'''", code, re.DOTALL)
    comments.extend(docstrings)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            function_names[node.name] += 1
            if ast.get_docstring(node):
                comments.append(ast.get_docstring(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    variable_names[target.id] += 1

    return (function_names, variable_names, comments), None


def process_comments(comments):
    word_freq = Counter()
    for comment in comments:
        words = re.findall(r"\b[a-zA-Z]+\b", comment.lower())
        word_freq.update(words)
    return word_freq


def scan_directory(base_directory, output_base):
    quarter_data = defaultdict(lambda: {"functions": Counter(), "variables": Counter(), "comments": []})
    skipped_files = defaultdict(list)

    for year_folder in tqdm(os.listdir(base_directory), desc="Top-level"):
        year_path = os.path.join(base_directory, year_folder)
        if not os.path.isdir(year_path):
            continue

        for quarter_folder in os.listdir(year_path):
            quarter_path = os.path.join(year_path, quarter_folder)
            if not os.path.isdir(quarter_path):
                continue

            for project in os.listdir(quarter_path):
                project_path = os.path.join(quarter_path, project)
                if not os.path.isdir(project_path):
                    continue

                time_info_path = os.path.join(project_path, "time_info.txt")
                if not os.path.exists(time_info_path):
                    continue

                file_quarters = parse_time_info(time_info_path, project_path)

                for file_path, (year, quarter) in file_quarters.items():
                    if not file_path.endswith(".py") or not os.path.exists(file_path):
                        continue

                    result, error = extract_code_info(file_path)
                    quarter_key = f"{year}Q{quarter}"

                    if error:
                        skipped_files[quarter_key].append(f"Skipped {file_path}: {error}")
                        continue

                    functions, variables, comments = result
                    quarter_data[quarter_key]["functions"].update(functions)
                    quarter_data[quarter_key]["variables"].update(variables)
                    quarter_data[quarter_key]["comments"].extend(comments)

    # 写入每个季度的数据
    for quarter_key, data in quarter_data.items():
        year, quarter = quarter_key[:4], quarter_key[5:]
        output_dir = os.path.join(output_base, year, f"Q{quarter}")
        os.makedirs(output_dir, exist_ok=True)

        comment_words_freq = process_comments(data["comments"])

        save_to_csv(sorted(data["functions"].items(), key=lambda x: x[1], reverse=True),
                    os.path.join(output_dir, f"functions_{quarter_key}.csv"),
                    ["Function Name", "Frequency"])
        save_to_csv(sorted(data["variables"].items(), key=lambda x: x[1], reverse=True),
                    os.path.join(output_dir, f"variables_{quarter_key}.csv"),
                    ["Variable Name", "Frequency"])
        save_to_csv(sorted(comment_words_freq.items(), key=lambda x: x[1], reverse=True),
                    os.path.join(output_dir, f"comments_words_{quarter_key}.csv"),
                    ["Word", "Frequency"])

        # 跳过文件记录
        skipped_log = os.path.join(output_dir, "skipped_files.txt")
        with open(skipped_log, "w", encoding="utf-8") as log:
            for entry in skipped_files[quarter_key]:
                log.write(entry + "\n")

    return quarter_data


def save_to_csv(sorted_data, file_path, header):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, value in sorted_data:
            writer.writerow([key, value])


# 运行配置
base_directory = "LLM_code/arxiv_dataset"
output_directory = "LLM_code/output_by_quarter/by_file"
os.makedirs(output_directory, exist_ok=True)

scan_directory(base_directory, output_directory)

print("✅ Processing completed. Results saved to output_by_quarter/by_feature.")

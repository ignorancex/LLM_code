import os
import ast
import re
import csv
from collections import defaultdict
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
                mapping[full_path] = f"{year}Q{quarter}"
    return mapping


def extract_code_info(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        if "\x00" in code:
            return set(), set()
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError):
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


def scan_directory(base_dir, output_base):
    function_quarter_map = defaultdict(lambda: defaultdict(set))
    variable_quarter_map = defaultdict(lambda: defaultdict(set))

    for year_folder in tqdm(os.listdir(base_dir), desc="Top-level"):
        year_path = os.path.join(base_dir, year_folder)
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

                for file_path, quarter in file_quarters.items():
                    if not file_path.endswith(".py") or not os.path.exists(file_path):
                        continue

                    functions, variables = extract_code_info(file_path)

                    for fn in functions:
                        function_quarter_map[fn][quarter].add(file_path)
                    for var in variables:
                        variable_quarter_map[var][quarter].add(file_path)

    all_quarters = sorted({q for name_map in (function_quarter_map, variable_quarter_map)
                           for v in name_map.values() for q in v})

    save_table_csv(function_quarter_map, all_quarters,
                   os.path.join(output_base, "functions_by_mod.csv"))

    save_table_csv(variable_quarter_map, all_quarters,
                   os.path.join(output_base, "variables_by_mod.csv"))


def save_table_csv(name_quarter_map, all_quarters, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name"] + all_quarters)
        for name, quarter_dict in sorted(name_quarter_map.items()):
            row = [name]
            for q in all_quarters:
                row.append(len(quarter_dict.get(q, set())))
            writer.writerow(row)


# === 主程序 ===
base_dir = "LLM_code/arxiv_dataset"
output_base = "LLM_code/output_by_quarter/by_mod/raw_data"
os.makedirs(output_base, exist_ok=True)

scan_directory(base_dir, output_base)

print("\n✅ Done. Saved as 'functions_by_mod.csv' and 'variables_by_mod.csv'")

import os
import csv
import re
import json
from collections import Counter
from tqdm import tqdm

"""
该脚本处理按文件的更新时间(精确到季度)组织的 Python 文件,
提取文件名基础部分的频率和注释行数,
然后为每个 "年+季度" 生成一个包含文件名频率的 CSV 文件,
并把统计信息（年份、季度、文件数、注释行数）输出到 JSON 文件中。

输出示例目录结构:
LLM_code/
  └─ output_by_quarter/
       └─ by_file/
           └─ 2020/
               └─ Q1/
                   └─ file_name_frequency_2020Q1.csv
       └─ summary.json
"""

def determine_quarter(year, month):
    """
    根据 year, month 判断属于哪个季度(Q1~Q4)，并将其限制在 [2020Q1, 2025Q1].
    其中 2025 年只有 Q1，其他月份也都视为 2025Q1。
    """
    # 先根据年份范围进行裁剪
    if year < 2020:
        return (2020, 1)   # 不足 2020 年的都归到 2020Q1
    if year > 2025:
        return (2025, 1)   # 超过 2025 年的都归到 2025Q1
    
    # 若在 2025 年内，则全部算作 Q1
    if year == 2025:
        return (2025, 1)
    
    # 对于 [2020, 2024] 年份，根据月份判定季度
    if 1 <= month <= 3:
        quarter = 1
    elif 4 <= month <= 6:
        quarter = 2
    elif 7 <= month <= 9:
        quarter = 3
    else:
        quarter = 4
    
    return (year, quarter)


def parse_time_info(file_path):
    """
    解析 time_info.txt 文件，提取 Python 文件路径和更新时间(精确到季度)
    返回一个字典 { (year, quarter): [文件路径, ...], ... }
    假设 time_info.txt 每行格式类似: "xxx.py: 2021-03-15"
    """
    quarter_files = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.match(r'(.+\.py):\s*(\d{4})-(\d{2})-(\d{2})', line.strip())
            if match:
                py_path, y, m, d = match.groups()
                year = int(y)
                month = int(m)
                # 根据日期判断对应季度
                final_year, final_quarter = determine_quarter(year, month)
                quarter_files.setdefault((final_year, final_quarter), []).append(py_path)
    return quarter_files


def scan_github_code(directory):
    """
    遍历 github_code 目录下的所有年份文件夹和季度子文件夹，
    并读取各项目的 time_info.txt，聚合所有 Python 文件对应的 (year, quarter) 信息。
    返回结构:
    {
      (year, quarter): [py_file_path1, py_file_path2, ...],
      ...
    }
    """
    all_quarter_py_files = {}

    # 获取所有年份文件夹
    year_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for year_folder in tqdm(year_folders, desc="Scanning github_code"):
        year_path = os.path.join(directory, year_folder)

        # 遍历 Q1~Q4 文件夹
        quarter_folders = [q for q in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, q))]
        for quarter_folder in quarter_folders:
            quarter_path = os.path.join(year_path, quarter_folder)

            # 遍历项目文件夹
            for project in os.listdir(quarter_path):
                project_path = os.path.join(quarter_path, project)
                time_info_path = os.path.join(project_path, 'time_info.txt')

                if os.path.exists(time_info_path):
                    quarter_files = parse_time_info(time_info_path)
                    for (y, q), file_list in quarter_files.items():
                        all_quarter_py_files.setdefault((y, q), []).extend(
                            [os.path.join(project_path, f) for f in file_list]
                        )

    return all_quarter_py_files



def extract_base_name(files):
    """提取文件名（去除扩展名）"""
    res = []
    for file in files:
        filename = os.path.basename(file)
        base, _ = os.path.splitext(filename)
        res.append(base)
    return res


def count_comments(files):
    """统计文件中的注释行数(以 # 开头的行为注释)"""
    total_comments = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                total_comments += sum(1 for line in f if line.strip().startswith('#'))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_comments


def write_word_frequency_to_csv(word_counter, output_file):
    """将文件名词频统计结果写入 CSV 文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Frequency'])
        for word, count in word_counter.most_common():
            writer.writerow([word, count])


def main(github_code_dir, output_dir):
    """
    遍历 github_code_dir 下的所有项目及其 time_info.txt，
    并按季度聚合输出结果到 output_dir.
    最后把统计信息输出到 JSON 文件中。
    """
    quarter_py_files = scan_github_code(github_code_dir)

    # 存储最终统计信息的列表
    summary_data = []

    # 这里对 quarter_py_files 用 tqdm 做进度展示：
    for (year, quarter), files in tqdm(quarter_py_files.items(), desc="Processing Quarters"):
        base_names = extract_base_name(files)
        word_counter = Counter(base_names)
        total_comments = count_comments(files)

        quarter_folder = f"Q{quarter}"
        # file_name_frequency_2020Q1.csv
        output_csv = os.path.join(
            output_dir,
            str(year),
            quarter_folder,
            f"file_name_frequency_{year}Q{quarter}.csv"
        )
        write_word_frequency_to_csv(word_counter, output_csv)

        # 记录到 summary_data
        summary_data.append({
            "Year": year,
            "Quarter": quarter,
            "Files": len(files),
            "Comments": total_comments
        })

    # 将 summary_data 写入 JSON
    # 输出到根目录下一个 summary.json，可以根据需要更改位置
    summary_json_path = os.path.join(output_dir, "summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary_data, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 这里替换为实际路径
    github_code_dir = 'LLM_code/arxiv_dataset'
    output_dir = 'LLM_code/output_by_quarter/by_file'
    main(github_code_dir, output_dir)

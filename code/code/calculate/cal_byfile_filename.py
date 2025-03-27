import os
import csv
import re
from collections import Counter
from pathlib import Path

"""
该脚本处理按年份组织的Python文件,
提取文件名基础部分的频率和注释行数,
然后为每个年份生成一个包含文件名频率的CSV文件,
同时输出每个年份的文件和注释的统计信息。
"""

def parse_time_info(file_path):
    """
    解析 time_info.txt 文件，提取 Python 文件路径和更新时间
    返回一个字典 {年份: [(文件路径, 年份)]}
    """
    year_files = {}

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.match(r'(.+\.py): (\d{4})-', line.strip())  # 提取文件路径和年份
            if match:
                file_path, year = match.groups()
                year_files.setdefault(year, []).append(file_path)

    return year_files


def scan_github_code(directory):
    """
    遍历 github_code 目录下的所有年份文件夹，并读取各项目的 time_info.txt
    返回一个 {年份: [Python 文件路径]} 的字典
    """
    year_py_files = {}

    for year_folder in os.listdir(directory):
        year_path = os.path.join(directory, year_folder)
        if not os.path.isdir(year_path):  # 确保是文件夹
            continue

        for project in os.listdir(year_path):  # 遍历项目
            project_path = os.path.join(year_path, project)
            time_info_path = os.path.join(project_path, 'time_info.txt')

            if os.path.exists(time_info_path):  # 如果存在 time_info.txt
                year_files = parse_time_info(time_info_path)
                for year, files in year_files.items():
                    year_py_files.setdefault(year, []).extend(
                        [os.path.join(project_path, f) for f in files]
                    )

    return year_py_files


def extract_base_name(files):
    """ 提取文件名（去除扩展名） """
    return [os.path.splitext(os.path.basename(file))[0] for file in files]


def count_comments(files):
    """ 统计文件中的注释行数 """
    total_comments = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                total_comments += sum(1 for line in f if line.strip().startswith('#'))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_comments


def write_word_frequency_to_csv(word_counter, output_file):
    """ 将文件名词频统计结果写入 CSV """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Frequency'])
        for word, count in word_counter.most_common():
            writer.writerow([word, count])


def main(github_code_dir, output_dir):
    year_py_files = scan_github_code(github_code_dir)

    for year, files in year_py_files.items():
        base_names = extract_base_name(files)
        word_counter = Counter(base_names)
        total_comments = count_comments(files)

        output_csv = os.path.join(output_dir, year, f'file_name_frequency_{year}.csv')
        write_word_frequency_to_csv(word_counter, output_csv)

        print(f"Year: {year}, Files: {len(files)}, Comments: {total_comments}")


if __name__ == "__main__":
    github_code_dir = './github_code'  # 替换为实际路径
    output_dir = './output'  # 结果存放路径
    main(github_code_dir, output_dir)

import os
import csv
import json
from collections import Counter
from tqdm import tqdm


def scan_directory_for_py_files(directory):
    """获取该目录下所有 .py 文件路径"""
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files


def extract_base_name(files):
    """提取不带扩展名的文件名"""
    return [os.path.splitext(os.path.basename(file))[0] for file in files]


def count_word_frequency(base_names):
    """统计文件名的频率"""
    return Counter(base_names)


def count_comments(files):
    """统计所有 Python 文件中的注释行数量"""
    total_comments = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                total_comments += sum(1 for line in f if line.strip().startswith('#'))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_comments


def write_word_frequency_to_csv(word_counter, output_file):
    """将文件名频率写入 CSV"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Frequency'])
        for word, count in word_counter.most_common():
            writer.writerow([word, count])


def process_quarter_directory(directory, output_csv_path):
    py_files = scan_directory_for_py_files(directory)
    print(f"📁 {directory} - Total Python files: {len(py_files)}")
    if not py_files:
        return 0, 0

    base_names = extract_base_name(py_files)
    word_counter = count_word_frequency(base_names)
    total_comments = count_comments(py_files)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    write_word_frequency_to_csv(word_counter, output_csv_path)

    print(f"✅ Output saved to {output_csv_path}")
    print(f"📝 Total number of comments: {total_comments}")

    return len(py_files), total_comments


def main():
    base_dir = 'LLM_code/arxiv_dataset'
    output_base = 'LLM_code/output_by_quarter/by_project'
    os.makedirs(output_base, exist_ok=True)

    summary = {}

    for year in range(2020, 2026):
        max_quarter = 1 if year == 2025 else 4
        for q in range(1, max_quarter + 1):
            year_str = str(year)
            quarter_str = f"Q{q}"
            quarter_path = os.path.join(base_dir, year_str, quarter_str)

            if not os.path.isdir(quarter_path):
                continue

            print(f"\n🔍 Scanning {year_str}/{quarter_str} ...")
            output_csv = os.path.join(output_base, year_str, quarter_str, "file_name_frequency.csv")
            total_files, total_comments = process_quarter_directory(quarter_path, output_csv)

            summary[f"{year_str}_{quarter_str}"] = {
                "total_py_files": total_files,
                "total_comments": total_comments
            }

    # 保存 summary 为 JSON
    summary_path = os.path.join(output_base, "stats_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\n📊 Summary saved to {summary_path}")
    print("🎉 All quarters processed successfully.")


if __name__ == "__main__":
    main()

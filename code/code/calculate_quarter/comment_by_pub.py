import os
import csv
import json
import re
import tokenize
from collections import Counter, defaultdict
import pandas as pd


def load_top_words(filepath):
    """加载前10000高频词为一个集合"""
    df = pd.read_csv(filepath)
    return set(df['word'].str.lower().tolist())


def scan_directory_for_py_files(directory):
    """获取该目录下所有 .py 文件路径"""
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files


def count_comments_word_freq_and_density(files, top_words_set):
    """统计注释行数、词频，并计算平均注释密度"""
    total_comments = 0
    word_counter = Counter()
    comment_ratios = []

    for file in files:
        comment_lines = 0
        total_lines = 0

        try:
            # 统计注释行并提取词频
            with open(file, 'rb') as f:
                tokens = tokenize.tokenize(f.readline)
                for toknum, tokval, _, _, _ in tokens:
                    if toknum == tokenize.COMMENT:
                        comment_lines += 1
                        comment_text = tokval.lstrip('#').strip().lower()
                        words = re.findall(r'\b[a-zA-Z]+\b', comment_text)
                        # 只统计在top_words_set中的词
                        filtered_words = [w for w in words if w in top_words_set]
                        word_counter.update(filtered_words)

            # 统计非空总行数
            with open(file, 'r', encoding='utf-8', errors='ignore') as f2:
                total_lines = sum(1 for line in f2 if line.strip())

            if total_lines > 0:
                comment_ratios.append(comment_lines / total_lines)

            total_comments += comment_lines

        except Exception as e:
            print(f"Error processing {file}: {e}")

    avg_comment_density = sum(comment_ratios) / len(comment_ratios) if comment_ratios else 0
    return total_comments, word_counter, avg_comment_density


def process_quarter_directory(directory, top_words_set):
    """处理单个季度目录，返回 Python 文件数、注释行数、注释密度和词频"""
    py_files = scan_directory_for_py_files(directory)
    print(f"📁 {directory} - Total Python files: {len(py_files)}")
    if not py_files:
        return 0, 0, 0.0, {}

    total_comments, word_counter, avg_comment_density = count_comments_word_freq_and_density(py_files, top_words_set)

    print(f"📝 Total comment lines: {total_comments}")
    print(f"📐 Average comment density: {avg_comment_density:.4f}")

    return len(py_files), total_comments, avg_comment_density, word_counter


def main():
    base_dir = 'LLM_code/arxiv_dataset'
    output_base = 'LLM_code/output_by_quarter/by_pub'
    os.makedirs(output_base, exist_ok=True)

    # 加载 top 10000 高频词集合
    top_words_file = 'top_words.csv'
    top_words_set = load_top_words(top_words_file)

    summary = {}
    all_quarter_word_counts = defaultdict(lambda: defaultdict(int))  # word -> quarter -> count
    all_quarters = []

    for year in range(2020, 2026):
        max_quarter = 1 if year == 2025 else 4
        for q in range(1, max_quarter + 1):
            year_str = str(year)
            quarter_str = f"Q{q}"
            quarter_path = os.path.join(base_dir, year_str, quarter_str)
            quarter_id = f"{year_str}_{quarter_str}"

            if not os.path.isdir(quarter_path):
                continue

            print(f"\n🔍 Scanning {quarter_id} ...")
            total_files, total_comments, avg_comment_density, word_counter = process_quarter_directory(quarter_path, top_words_set)

            # 汇总词频（每个词 -> 当前季度的频次）
            for word, freq in word_counter.items():
                all_quarter_word_counts[word][quarter_id] += freq

            all_quarters.append(quarter_id)

            # 保存 summary 信息
            summary[quarter_id] = {
                "total_py_files": total_files,
                "total_comments": total_comments,
                "avg_comment_density": round(avg_comment_density, 4)
            }

    # 去重并排序季度列
    all_quarters = sorted(set(all_quarters))

    # 写入总词频 CSV：每一行一个 word，每一列是季度频次
    wordfreq_path = os.path.join(output_base, "all_quarters_word_frequency_new.csv")
    with open(wordfreq_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Word'] + all_quarters
        writer.writerow(header)

        for word in sorted(all_quarter_word_counts.keys()):
            row = [word]
            for q in all_quarters:
                row.append(all_quarter_word_counts[word].get(q, 0))
            writer.writerow(row)

    # 写入 summary JSON
    summary_path = os.path.join(output_base, "comment_ratio.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\n📊 Summary saved to {summary_path}")
    print(f"🧾 Word frequency saved to {wordfreq_path}")
    print("🎉 All quarters processed successfully.")


if __name__ == "__main__":
    main()

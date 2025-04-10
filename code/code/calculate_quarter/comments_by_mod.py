import os
import csv
import json
import re
import tokenize
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm


def load_top_words(filepath):
    """加载前10000高频词为一个集合"""
    df = pd.read_csv(filepath)
    return set(df['word'].str.lower().tolist())


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
                mapping[full_path] = f"{year}_Q{quarter}"
    return mapping


def count_comments_word_freq_and_density(file_path, top_words_set):
    """统计单个文件的注释词频、密度"""
    comment_lines = 0
    total_lines = 0
    word_counter = Counter()

    try:
        with open(file_path, 'rb') as f:
            tokens = tokenize.tokenize(f.readline)
            for toknum, tokval, _, _, _ in tokens:
                if toknum == tokenize.COMMENT:
                    comment_lines += 1
                    comment_text = tokval.lstrip('#').strip().lower()
                    words = re.findall(r'\b[a-zA-Z]+\b', comment_text)
                    filtered_words = [w for w in words if w in top_words_set]
                    word_counter.update(filtered_words)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f2:
            total_lines = sum(1 for line in f2 if line.strip())

        ratio = comment_lines / total_lines if total_lines > 0 else 0
        return comment_lines, word_counter, ratio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, Counter(), 0.0


def main():
    base_dir = 'LLM_code/arxiv_dataset'
    output_base = 'LLM_code/output_by_quarter/by_mod'
    os.makedirs(output_base, exist_ok=True)

    top_words_file = 'top_words.csv'
    top_words_set = load_top_words(top_words_file)

    summary = defaultdict(lambda: {
        "total_py_files": 0,
        "total_comments": 0,
        "avg_comment_density": 0.0
    })
    word_freq_by_quarter = defaultdict(lambda: defaultdict(int))
    density_accumulator = defaultdict(list)

    for year_folder in tqdm(os.listdir(base_dir), desc="Scanning base_dir"):
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

                time_info_path = os.path.join(project_path, 'time_info.txt')
                if not os.path.exists(time_info_path):
                    continue

                file_quarters = parse_time_info(time_info_path, project_path)

                for file_path, quarter in file_quarters.items():
                    if not file_path.endswith(".py") or not os.path.exists(file_path):
                        continue

                    comment_lines, word_counter, density = count_comments_word_freq_and_density(file_path, top_words_set)

                    summary[quarter]["total_py_files"] += 1
                    summary[quarter]["total_comments"] += comment_lines
                    density_accumulator[quarter].append(density)

                    for word, count in word_counter.items():
                        word_freq_by_quarter[word][quarter] += count

    # 写入 summary json
    for q in summary:
        dlist = density_accumulator[q]
        summary[q]["avg_comment_density"] = round(sum(dlist) / len(dlist), 4) if dlist else 0.0

    summary_path = os.path.join(output_base, "comment_ratio.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    # 写入词频 CSV
    all_quarters = sorted(set(q for counts in word_freq_by_quarter.values() for q in counts))
    wordfreq_path = os.path.join(output_base, "word_frequency_10000.csv")
    with open(wordfreq_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word'] + all_quarters)
        for word in sorted(word_freq_by_quarter.keys()):
            row = [word] + [word_freq_by_quarter[word].get(q, 0) for q in all_quarters]
            writer.writerow(row)

    print(f"\n📊 Summary saved to {summary_path}")
    print(f"🧾 Word frequency saved to {wordfreq_path}")
    print("🎉 All projects processed successfully.")


if __name__ == "__main__":
    main()

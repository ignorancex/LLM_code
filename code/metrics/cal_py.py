import os
import csv
import json
import concurrent.futures
from radon.metrics import h_visit, mi_visit, mi_parameters, mi_compute
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

MAX_FILE_SIZE = int(0.02 * 1024 * 1024)  # 1MB in bytes

def analyze_py(path: str):
    """分析单个 Python 文件，返回复杂度指标。"""
    try:
        code = open(path, encoding='utf-8', errors='ignore').read()
        hal = h_visit(code)
        t = hal.total
        mi_std = mi_visit(code, True)
        hal_vol, cyclo, sloc, comment_rate = mi_parameters(code)
        mi_custom = mi_compute(hal_vol, cyclo, sloc, comment_rate)

        return {
            'h1': t.h1, 'h2': t.h2, 'N1': t.N1, 'N2': t.N2,
            'vocabulary': t.vocabulary, 'length': t.length,
            'calculated_length': t.calculated_length, 'volume': t.volume,
            'difficulty': t.difficulty, 'effort': t.effort,
            'time_sec': t.time, 'bugs': t.bugs,
            'cyclomatic': cyclo, 'sloc': sloc, 'lloc': sloc,
            'comment_rate': comment_rate, 'mi_std': mi_std, 'mi_custom': mi_custom
        }
    except Exception:
        return None

def average_dicts(dicts):
    if not dicts:
        return None
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}

def load_quarter_repo_category(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    quarter_repo_category = defaultdict(dict)
    for quarter, items in raw.items():
        for item in items:
            link = item["link"]
            category = item["categories"]
            repo = link.rstrip("/").split("/")[-1]
            quarter_repo_category[quarter][repo] = 'cs' if category.startswith("cs.") else 'non_cs'
    return quarter_repo_category

def main(
    root='LLM_code/arxiv_dataset',
    category_json='LLM_code/code/github_links/python_dataset_links_1.json',
    out_csv='github_py_metrics_by_category.csv'
):
    fields = [
        'quarter', 'category',
        'h1','h2','N1','N2','vocabulary','length','calculated_length',
        'volume','difficulty','effort','time_sec','bugs',
        'cyclomatic','sloc','lloc','comment_rate','mi_std','mi_custom'
    ]
    rows = []

    quarter_repo_category = load_quarter_repo_category(category_json)

    for year in range(2020, 2026):
        quarters = ['Q1', 'Q2', 'Q3', 'Q4'] if year < 2025 else ['Q1']
        for q in quarters:
            quarter = f"{year}{q}"
            quarter_dir = os.path.join(root, str(year), q)
            if not os.path.isdir(quarter_dir):
                continue

            all_py_files = []
            repo_map = {}

            for repo in os.listdir(quarter_dir):
                repo_path = os.path.join(quarter_dir, repo)
                if not os.path.isdir(repo_path): continue
                for dirpath, _, files in os.walk(repo_path):
                    for fn in files:
                        if fn.endswith(".py"):
                            full_path = os.path.join(dirpath, fn)
                            all_py_files.append(full_path)
                            repo_map[full_path] = repo  # 记录该文件属于哪个repo

            total_count = len(all_py_files)
            skipped_count = 0
            category_metrics = defaultdict(list)

            print(f"🔍 Analyzing {quarter}: {total_count} .py files")

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                for path in all_py_files:
                    if os.path.getsize(path) > MAX_FILE_SIZE:
                        skipped_count += 1
                        continue
                    futures[executor.submit(analyze_py, path)] = path

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"[{quarter}]"):
                    path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            repo = repo_map.get(path, None)
                            cat = quarter_repo_category.get(quarter, {}).get(repo)
                            if cat in ['cs', 'non_cs']:
                                category_metrics[cat].append(result)
                    except Exception:
                        continue

            for cat in ['cs', 'non_cs']:
                avg = average_dicts(category_metrics[cat])
                if avg:
                    avg['quarter'] = quarter
                    avg['category'] = cat
                    rows.append(avg)

            print(f"📊 {quarter}: Processed {total_count - skipped_count}, Skipped {skipped_count} (>1MB files)\n")

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Done. 汇总写入 {out_csv}")

if __name__ == '__main__':
    main()

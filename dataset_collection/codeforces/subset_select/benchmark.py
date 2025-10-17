import json
import random
import re
from collections import defaultdict
import pandas as pd

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_info(item):
    tags = item.get('tags', [])
    difficulty = None
    algorithms = []
    for tag in tags:
        if re.match('^\\*\\d+$', tag):
            difficulty = int(tag[1:])
        else:
            algorithms.append(tag)
    return (difficulty, algorithms)

def get_difficulty_bucket(difficulty):
    if difficulty is None:
        return None
    if 800 <= difficulty <= 1199:
        return '800-1199'
    elif 1200 <= difficulty <= 1599:
        return '1200-1599'
    elif 1600 <= difficulty <= 1999:
        return '1600-1999'
    else:
        return '2000+'

def print_distribution(selected_data):
    stats = defaultdict(lambda : defaultdict(int))
    for item in selected_data:
        bucket = get_difficulty_bucket(item['difficulty'])
        alg = item['main_algorithm']
        stats[bucket][alg] += 1
    rows = []
    for (bucket, alg_counts) in stats.items():
        for (alg, count) in alg_counts.items():
            rows.append({'Difficulty_Bucket': bucket, 'Algorithm': alg, 'Count': count})
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['Difficulty_Bucket', 'Algorithm'])
    for bucket in ['800-1199', '1200-1599', '1600-1999', '2000+']:
        count = df[df['Difficulty_Bucket'] == bucket]['Count'].sum()

def build_benchmark(input_path, output_path, total_questions=200):
    data = load_jsonl(input_path)
    target_algorithms = ['implementation', 'brute force', 'constructive algorithms', 'greedy', 'binary search', 'math', 'dp', 'data structures', 'combinatorics', 'dfs and similar']
    bucket_questions = defaultdict(list)
    for item in data:
        (difficulty, algorithms) = extract_info(item)
        bucket = get_difficulty_bucket(difficulty)
        if bucket and algorithms:
            item['difficulty'] = difficulty
            item['main_algorithm'] = algorithms[0]
            bucket_questions[bucket].append(item)
    buckets = ['800-1199', '1200-1599', '1600-1999', '2000+']
    per_bucket = total_questions // len(buckets)
    selected = []
    for bucket in buckets:
        candidates = bucket_questions[bucket]
        alg_groups = defaultdict(list)
        for item in candidates:
            if item['main_algorithm'] in target_algorithms:
                alg_groups[item['main_algorithm']].append(item)
        total_in_bucket = sum((len(alg_groups[alg]) for alg in target_algorithms))
        for alg in target_algorithms:
            if len(alg_groups[alg]) == 0:
                raise ValueError(f"❌ Error")
        algo_quota = {}
        for alg in target_algorithms:
            ratio = len(alg_groups[alg]) / total_in_bucket
            num_to_pick = round(ratio * per_bucket)
            algo_quota[alg] = num_to_pick
        total_assigned = sum(algo_quota.values())
        if total_assigned != per_bucket:
            diff = per_bucket - total_assigned
            sorted_algs = sorted(target_algorithms, key=lambda x: -len(alg_groups[x]))
            for i in range(abs(diff)):
                adjust_alg = sorted_algs[i % len(sorted_algs)]
                algo_quota[adjust_alg] += 1 if diff > 0 else -1
        bucket_selected = []
        for alg in target_algorithms:
            items = alg_groups[alg]
            if len(items) < algo_quota[alg]:
                raise ValueError(f"❌ Error")
            bucket_selected.extend(random.sample(items, algo_quota[alg]))
        selected.extend(bucket_selected)
    save_jsonl(selected, output_path)
    print_distribution(selected)
build_benchmark('LLM_code/codeforces/subset_select/intersection.jsonl', 'LLM_code/codeforces/subset_select/benchmark_200.jsonl')
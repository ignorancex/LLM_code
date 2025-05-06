import json
import random
import re
from collections import defaultdict
import pandas as pd

# 读取JSONL文件
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# 保存成JSONL文件
def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 提取difficulty和算法
def extract_info(item):
    tags = item.get('tags', [])
    difficulty = None
    algorithms = []
    for tag in tags:
        if re.match(r'^\*\d+$', tag):
            difficulty = int(tag[1:])
        else:
            algorithms.append(tag)
    return difficulty, algorithms

# 难度划分
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

# 打印分布
def print_distribution(selected_data):
    stats = defaultdict(lambda: defaultdict(int))

    for item in selected_data:
        bucket = get_difficulty_bucket(item['difficulty'])
        alg = item['main_algorithm']
        stats[bucket][alg] += 1

    rows = []
    for bucket, alg_counts in stats.items():
        for alg, count in alg_counts.items():
            rows.append({
                "Difficulty_Bucket": bucket,
                "Algorithm": alg,
                "Count": count
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Difficulty_Bucket", "Algorithm"])

    print("\n📊 各桶各算法类别统计：")
    print(df.to_string(index=False))

    print("\n✅ 每个难度桶的总数：")
    for bucket in ['800-1199', '1200-1599', '1600-1999', '2000+']:
        count = df[df['Difficulty_Bucket'] == bucket]['Count'].sum()
        print(f"{bucket}: {count} 题")

# 主程序
def build_benchmark(input_path, output_path, total_questions=200):
    data = load_jsonl(input_path)
    print(f"总共载入 {len(data)} 道题目")

    # 指定希望每个桶都出现的算法列表
    target_algorithms = ['implementation', 'brute force', 
    'constructive algorithms', 'greedy',
    'binary search', 'math', 
    'dp', 'data structures', 
    'combinatorics', 'dfs and similar']

    bucket_questions = defaultdict(list)
    for item in data:
        difficulty, algorithms = extract_info(item)
        bucket = get_difficulty_bucket(difficulty)
        if bucket and algorithms:
            item['difficulty'] = difficulty
            item['main_algorithm'] = algorithms[0]
            bucket_questions[bucket].append(item)

    buckets = ['800-1199', '1200-1599', '1600-1999', '2000+']
    per_bucket = total_questions // len(buckets)  # 每桶50题

    selected = []

    for bucket in buckets:
        candidates = bucket_questions[bucket]
        print(f"\n{bucket} 区间有 {len(candidates)} 道题目可选")

        # 分成算法组
        alg_groups = defaultdict(list)
        for item in candidates:
            if item['main_algorithm'] in target_algorithms:
                alg_groups[item['main_algorithm']].append(item)

        # 计算总量
        total_in_bucket = sum(len(alg_groups[alg]) for alg in target_algorithms)

        # 检查是否有类别缺失
        for alg in target_algorithms:
            if len(alg_groups[alg]) == 0:
                raise ValueError(f"❌ Error: {bucket} 区间缺少算法类别 '{alg}'，无法满足要求。")

        # 计算每种算法实际要取多少题（按比例）
        algo_quota = {}
        for alg in target_algorithms:
            ratio = len(alg_groups[alg]) / total_in_bucket
            num_to_pick = round(ratio * per_bucket)
            algo_quota[alg] = num_to_pick

        # 微调：防止四舍五入导致加起来不是50
        total_assigned = sum(algo_quota.values())
        if total_assigned != per_bucket:
            diff = per_bucket - total_assigned
            print(f"⚙️ 调整配额（差异 {diff} 题）")
            sorted_algs = sorted(target_algorithms, key=lambda x: -len(alg_groups[x]))
            for i in range(abs(diff)):
                adjust_alg = sorted_algs[i % len(sorted_algs)]
                algo_quota[adjust_alg] += 1 if diff > 0 else -1

        # 按计算好的数量随机抽取
        bucket_selected = []
        for alg in target_algorithms:
            items = alg_groups[alg]
            if len(items) < algo_quota[alg]:
                raise ValueError(f"❌ Error: {bucket} 区间算法 '{alg}' 可选题量不足，只有 {len(items)}，要求 {algo_quota[alg]}")
            bucket_selected.extend(random.sample(items, algo_quota[alg]))

        selected.extend(bucket_selected)

    save_jsonl(selected, output_path)
    print(f"\n✅ 成功选取 {len(selected)} 道题目，保存至 {output_path}")

    print_distribution(selected)

# 使用示例
build_benchmark(
    'LLM_code/codeforces/subset_select/intersection.jsonl', 
    'LLM_code/codeforces/subset_select/benchmark_200.jsonl'
)

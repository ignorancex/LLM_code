import json
import re
from collections import defaultdict

# 读取JSONL文件
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

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

# 难度桶划分
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

# 主程序
def analyze_distribution_by_bucket(input_path):
    data = load_jsonl(input_path)
    print(f"✅ 总共载入 {len(data)} 道题目")

    algo_counter = defaultdict(int)
    bucket_algo_counter = defaultdict(lambda: defaultdict(int))

    for item in data:
        difficulty, algorithms = extract_info(item)
        if not algorithms:
            continue
        main_alg = algorithms[0]
        # 排除特殊标签
        if main_alg == '*special problem':
            continue

        algo_counter[main_alg] += 1

        bucket = get_difficulty_bucket(difficulty)
        if bucket:
            bucket_algo_counter[bucket][main_alg] += 1

    # 选出出现次数最多的前10个算法
    sorted_algos = sorted(algo_counter.items(), key=lambda x: x[1], reverse=True)
    top10_algorithms = [alg for alg, _ in sorted_algos[:10]]

    print("\n📊 Top10算法类别：")
    for i, alg in enumerate(top10_algorithms, 1):
        print(f"{i}. {alg}")

    # 统计每个桶里Top10算法的数量
    print("\n📋 每个难度桶内Top10算法分布：")
    header = f"{'Algorithm':<25} {'800-1199':>10} {'1200-1599':>10} {'1600-1999':>10} {'2000+':>10}"
    print(header)
    print("-" * len(header))

    for alg in top10_algorithms:
        row = f"{alg:<25}"
        for bucket in ['800-1199', '1200-1599', '1600-1999', '2000+']:
            count = bucket_algo_counter[bucket].get(alg, 0)
            row += f"{count:>10}"
        print(row)

# 使用示例
analyze_distribution_by_bucket('LLM_code/codeforces/subset_select/intersection.jsonl')

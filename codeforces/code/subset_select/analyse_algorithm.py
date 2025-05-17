import json
from collections import defaultdict

# 定义桶映射函数
def get_difficulty_bucket(difficulty):
    if 800 <= difficulty <= 1199:
        return '800-1199'
    elif 1200 <= difficulty <= 1599:
        return '1200-1599'
    elif 1600 <= difficulty <= 1999:
        return '1600-1999'
    elif difficulty >= 2000:
        return '2000+'
    else:
        return None  # 忽略低于800的题目（如无效难度）

# 输入文件路径
input_path = "LLM_code/codeforces/subset_select/benchmark_200.jsonl"

# 初始化计数字典
stats = defaultdict(lambda: defaultdict(int))

# 逐行读取并计数
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        difficulty = data.get("difficulty")
        algorithm = data.get("main_algorithm")

        if difficulty is None or algorithm is None:
            continue

        bucket = get_difficulty_bucket(difficulty)
        if bucket:
            stats[bucket][algorithm] += 1

# 打印结果
for bucket in ['800-1199', '1200-1599', '1600-1999', '2000+']:
    print(f"Difficulty Bucket: {bucket}")
    for algo, count in sorted(stats[bucket].items(), key=lambda x: -x[1]):
        print(f"  {algo}: {count}")

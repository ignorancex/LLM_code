import json

# 读取 JSON 文件（list）
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取 JSONL 文件（benchmark问题名单）
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# 保存成 JSONL
def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 主逻辑
def extract_context(input_full_data_path, benchmark_path, output_path):
    # 加载大数据集
    full_data = load_json(input_full_data_path)

    # 加载benchmark的problem名字
    benchmark_data = load_jsonl(benchmark_path)
    benchmark_problems = set(item['problem'] for item in benchmark_data)
    print(f"⭐ Benchmark问题数量：{len(benchmark_problems)}")

    # 筛选
    selected = []
    for item in full_data:
        if item['fullname'] in benchmark_problems:
            selected.append({
                "problem": item['fullname'],
                "context": item['context'],
                "context_plain": item['context_plain']
            })

    # 保存
    save_jsonl(selected, output_path)
    print(f"✅ 成功提取 {len(selected)} 道题目，保存至 {output_path}")

# 使用示例
extract_context('dataset/unique_problem_python.json', 
'LLM_code/codeforces/subset_select/benchmark_200.jsonl', 
'LLM_code/codeforces/subset_select/benchmark.jsonl')

import json

# 读取JSONL文件
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

# 保存成JSONL文件
def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 主逻辑
def merge_common_problems(file1_path, file2_path, output_path):
    # 加载两个文件
    data1 = load_jsonl(file1_path)
    data2 = load_jsonl(file2_path)

    # 建立problem到条目的映射
    dict1 = {item['problem']: item for item in data1}
    dict2 = {item['problem']: item for item in data2}

    # 找交集
    common_problems = set(dict1.keys()) & set(dict2.keys())

    # 合并交集部分
    merged_data = []
    for problem in common_problems:
        merged_data.append(dict1[problem])

    # 保存
    save_jsonl(merged_data, output_path)
    print(f"✅ 合并完成，共 {len(merged_data)} 条记录，保存到 {output_path}")

# 示例使用
merge_common_problems('LLM_code/codeforces/label/tags_cpp.jsonl', 
                        'LLM_code/codeforces/label/tags_py.jsonl', 
                        'LLM_code/codeforces/subset_select/intersection.jsonl')

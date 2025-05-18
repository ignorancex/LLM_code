import json

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_common_problems(file1_path, file2_path, output_path):
    data1 = load_jsonl(file1_path)
    data2 = load_jsonl(file2_path)
    dict1 = {item['problem']: item for item in data1}
    dict2 = {item['problem']: item for item in data2}
    common_problems = set(dict1.keys()) & set(dict2.keys())
    merged_data = []
    for problem in common_problems:
        merged_data.append(dict1[problem])
    save_jsonl(merged_data, output_path)
merge_common_problems('LLM_code/codeforces/label/tags_cpp.jsonl', 'LLM_code/codeforces/label/tags_py.jsonl', 'LLM_code/codeforces/subset_select/intersection.jsonl')
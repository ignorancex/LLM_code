import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

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

def extract_context(input_full_data_path, benchmark_path, output_path):
    full_data = load_json(input_full_data_path)
    benchmark_data = load_jsonl(benchmark_path)
    benchmark_problems = set((item['problem'] for item in benchmark_data))
    selected = []
    for item in full_data:
        if item['fullname'] in benchmark_problems:
            selected.append({'problem': item['fullname'], 'context_plain': item['context_plain']})
    save_jsonl(selected, output_path)
extract_context('dataset/unique_problem_python.json', 'LLM_code/codeforces/subset_select/benchmark_200.jsonl', 'LLM_code/codeforces/subset_select/benchmark.jsonl')
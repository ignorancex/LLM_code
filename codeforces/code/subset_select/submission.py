import json
from collections import defaultdict
benchmark_path = 'LLM_code/codeforces/subset_select/benchmark.jsonl'
problem_ids = set()
with open(benchmark_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            problem_ids.add(obj.get('problem'))
cf_data_path = 'dataset/cf_python_plain.json'
problem_submission_count = defaultdict(int)
with open(cf_data_path, 'r', encoding='utf-8') as f:
    submissions = json.load(f)
    for sub in submissions:
        pid = sub.get('fullname')
        if pid in problem_ids:
            problem_submission_count[pid] += 1
if problem_submission_count:
    min_count = min(problem_submission_count.values())
    min_problems = [pid for (pid, cnt) in problem_submission_count.items() if cnt == min_count]
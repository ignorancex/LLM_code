import json
from tqdm import tqdm
problem_path = 'LLM_code/codeforces/problem_plaintext.json'
submission_path = 'LLM_code/codeforces/cf_code.json'
output_path = 'LLM_code/codeforces/matched_submissions_with_problem.json'
with open(problem_path, 'r', encoding='utf-8') as f:
    problems = json.load(f)
with open(submission_path, 'r', encoding='utf-8') as f:
    submissions = json.load(f)
submissions_by_problem = {}
for sub in submissions:
    pid = sub['problems_id']
    submissions_by_problem.setdefault(pid, []).append(sub)
merged_output = []
for problem in tqdm(problems, desc='Merging problems with submissions'):
    pid = problem['id']
    if pid in submissions_by_problem:
        submission = submissions_by_problem[pid][0]
        merged = problem.copy()
        merged.update(submission)
        merged_output.append(merged)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_output, f, ensure_ascii=False, indent=2)
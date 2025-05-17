import json
from tqdm import tqdm

# 输入文件路径
problem_path = "LLM_code/codeforces/problem_plaintext.json"
submission_path = "LLM_code/codeforces/cf_code.json"
output_path = "LLM_code/codeforces/matched_submissions_with_problem.json"

# 读取题目信息
with open(problem_path, "r", encoding="utf-8") as f:
    problems = json.load(f)

# 读取提交信息
with open(submission_path, "r", encoding="utf-8") as f:
    submissions = json.load(f)

# 根据 problems_id 建立提交映射
submissions_by_problem = {}
for sub in submissions:
    pid = sub["problems_id"]
    submissions_by_problem.setdefault(pid, []).append(sub)

# 生成合并后的输出
merged_output = []
for problem in tqdm(problems, desc="Merging problems with submissions"):
    pid = problem["id"]
    if pid in submissions_by_problem:
        submission = submissions_by_problem[pid][0]  # 可改为其他策略
        # 合并题目信息与提交信息
        merged = problem.copy()
        merged.update(submission)
        merged_output.append(merged)

# 写入输出文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_output, f, ensure_ascii=False, indent=2)

print(f"✅ 合并完成，保存到 {output_path}")

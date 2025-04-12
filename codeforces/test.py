import re

def remove_control_characters(json_string):
    # 删除除 \n, \t 之外的所有控制字符（ASCII < 32）
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_string)

input_path = "/media/sata3/siming/LLM_code/codeforces/problem.json"
output_path = "/media/sata3/siming/LLM_code/codeforces/problem_cleaned.json"

with open(input_path, "r", encoding="utf-8") as f:
    raw = f.read()

cleaned = remove_control_characters(raw)

# 验证是否能成功解析为 JSON
import json
try:
    parsed = json.loads(cleaned)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print("✅ 清理成功，保存到:", output_path)
except json.JSONDecodeError as e:
    print("❌ 清理失败，仍然有错误:", e)

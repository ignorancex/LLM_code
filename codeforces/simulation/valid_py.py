import json
import ast
from pathlib import Path

# 输入输出路径
input_path = Path("LLM_code/codeforces/simulation/qwen_32b_cpp_extract.json")
output_path = Path("LLM_code/codeforces/simulation/valid/qwen_32b_cpp_valid.json")

# 加载数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 检查 Python 代码是否合法
def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

# 过滤数据
valid_data = []
for item in data:
    code_block = item.get("generate_code_block", "").strip()
    ref_block = item.get("generate_ref_code_block", "").strip()

    if code_block and ref_block:
        if is_valid_python(code_block) and is_valid_python(ref_block):
            valid_data.append(item)

# 保存新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)

print(f"Filtered valid entries saved to: {output_path}")
print(f"Total valid entries: {len(valid_data)}")

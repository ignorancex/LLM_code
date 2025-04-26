import json
import re

def extract_reasoning_and_code(text):
    # 提取 Reasoning 块
    reasoning_match = re.search(r"### Reasoning\s*(.*?)\s*### Code", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # 提取 Code 块
    code_match = re.search(r"### Code\s*```[a-zA-Z]*\s*(.*?)```", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    return reasoning, code

# 读取原始 JSON 文件
with open("LLM_code/codeforces/simulation/deepseek_32b_cpp.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理每条记录
for item in data:
    gen = item.get("generate_code", "")
    ref = item.get("generate_code_ref", "")

    reasoning, code = extract_reasoning_and_code(gen)
    ref_reasoning, ref_code = extract_reasoning_and_code(ref)

    item["generate_reasoning"] = reasoning
    item["generate_code_block"] = code
    item["generate_ref_reasoning"] = ref_reasoning
    item["generate_ref_code_block"] = ref_code

# 保存到新的 JSON 文件
with open("LLM_code/codeforces/simulation/deepseek_32b_cpp_1.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

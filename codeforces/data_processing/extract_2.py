import json
import re

def extract_reasoning_and_code(text):
    # 提取 Reasoning 块
    reasoning_match = re.search(r"### Reasoning\s*(.*?)\s*(### Code|$)", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # 提取 Code 块
    code_match = re.search(r"### Code\s*(.*)", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    # 如果 code 块是 ``` 包裹的，去掉 ``` 包围
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\s*", "", code)  # 去掉开头的 ```
        code = re.sub(r"```$", "", code)  # 去掉结尾的 ```
        code = code.strip()

    return reasoning, code

# 读取原始 JSON 文件
with open("simulation/deepseek_32b_cpp_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理每条记录
for item in data:
    # 只有在 generate_code_block 为空时，才重新提取
    if not item.get("generate_code_block", "").strip():
        gen = item.get("generate_code", "")
        reasoning, code = extract_reasoning_and_code(gen)
        item["generate_reasoning"] = reasoning
        item["generate_code_block"] = code

    # 只有在 generate_ref_code_block 为空时，才重新提取
    if not item.get("generate_ref_code_block", "").strip():
        ref = item.get("generate_code_ref", "")
        ref_reasoning, ref_code = extract_reasoning_and_code(ref)
        item["generate_ref_reasoning"] = ref_reasoning
        item["generate_ref_code_block"] = ref_code

# 保存到新的 JSON 文件
with open("deepseek_32b_cpp_2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

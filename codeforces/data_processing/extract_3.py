import json
import re

def extract_cpp_block(text):
    """提取```cpp```包裹的内容"""
    cpp_match = re.search(r"```cpp\s*(.*?)\s*```", text, re.DOTALL)
    if cpp_match:
        return cpp_match.group(1).strip()
    return ""

# 读取原始 JSON 文件
with open("simulation/deepseek_32b_cpp_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理每条记录
for item in data:
    # 如果 generate_code_block 为空，只提取```cpp```代码
    if not item.get("generate_code_block", "").strip():
        gen = item.get("generate_code", "")
        cpp_code = extract_cpp_block(gen)
        if cpp_code:
            item["generate_code_block"] = cpp_code

    # 如果 generate_ref_code_block 为空，只提取```cpp```代码
    if not item.get("generate_ref_code_block", "").strip():
        ref = item.get("generate_code_ref", "")
        ref_cpp_code = extract_cpp_block(ref)
        if ref_cpp_code:
            item["generate_ref_code_block"] = ref_cpp_code

# 保存到新的 JSON 文件
with open("simulation/deepseek_32b_cpp_extract.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

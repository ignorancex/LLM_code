import json

# 读取你的 JSON 文件
with open('LLM_code/codeforces/simulation/deepseek_32b_cpp_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计数
empty_count = 0

for item in data:
    code = item.get('generate_code', '').strip()
    code_block = item.get('generate_code_block', '').strip()

    ref_code = item.get('generate_code_ref', '').strip()
    ref_code_block = item.get('generate_ref_code_block', '').strip()

    # 检查 generate_code_block
    if not code_block and code:
        empty_count += 1

    # 检查 generate_ref_code_block
    if not ref_code_block and ref_code:
        empty_count += 1

print(f"共有 {empty_count} 个 generate_code_block 或 generate_ref_code_block 为空（且原始存在）。")

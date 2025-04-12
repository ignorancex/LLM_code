import json

# 输入输出文件路径
input_file = 'LLM_code/codeforces/cf_code.json'
output_file = 'LLM_code/codeforces/cf_code_clean.json'

# 要搜索的敏感关键词
sensitive_token = '353388068:AAE-N_3Ic7rD8EMTv-wgofoBscJT_ofwbG4'

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f, strict=False)

# 过滤掉包含敏感内容的项
filtered_data = [
    item for item in data
    if sensitive_token not in item.get('sourceCode', '')
]

# 保存过滤后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"✅ 已完成过滤，剩余条目数：{len(filtered_data)}，已保存到：{output_file}")

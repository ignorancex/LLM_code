import json
from bs4 import BeautifulSoup
from tqdm import tqdm

# 输入和输出文件路径
input_path = "LLM_code/codeforces/cf_code_clean.json"
output_path = "LLM_code/codeforces/cf_code_plain.json"

# 读取 JSON 文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 批量转换
for item in tqdm(data, desc="Extracting text from HTML"):
    html = item.get("context", "")
    soup = BeautifulSoup(html, "html.parser")
    # 去除空行和多余空格
    text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    item["context_plain"] = text  # 新增字段保存纯文本

# 保存新 JSON 文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功提取纯文本，保存为 {output_path}")

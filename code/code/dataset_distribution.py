import json
import csv
from datetime import datetime
from collections import Counter

# === 路径设置 ===
combined_links_path = 'LLM_code/dataset/github_links/links_non_empty.json'
full_info_path = 'LLM_code/dataset/github_links/Only_links_set.json'
output_json_path = 'LLM_code/dataset/github_links/valid_links.json'
output_csv_path = 'LLM_code/dataset/github_links/quarterly_link_counts.csv'

# === 读取合并后的链接文件 ===
with open(combined_links_path, 'r', encoding='utf-8') as f:
    combined_data = json.load(f)
combined_links = set(combined_data.get("github_links", []))

# === 读取包含 update_date 的完整信息文件 ===
with open(full_info_path, 'r', encoding='utf-8') as f:
    full_info = json.load(f)

# === 匹配链接并提取更新时间 ===
matched_links_with_date = []
for item in full_info:
    link = item.get("github_links")
    if link in combined_links:
        matched_links_with_date.append({
            "github_links": link,
            "update_date": item.get("update_date")
        })

# === 保存为新的 JSON 文件 ===
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(matched_links_with_date, f, indent=4)

print(f"✅ 共匹配到 {len(matched_links_with_date)} 条链接，已保存到 {output_json_path}")

# === 统计每个季度的数量 ===

# 辅助函数：从日期生成季度标签，如 "2021Q3"
def get_quarter(dt):
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return f"{year}Q{quarter}"

# 设定时间范围
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 3, 31)

# 统计季度数量
quarter_counts = Counter()
for item in matched_links_with_date:
    try:
        date = datetime.strptime(item["update_date"], "%Y-%m-%d")
        if start_date <= date <= end_date:
            quarter = get_quarter(date)
            quarter_counts[quarter] += 1
    except Exception as e:
        print(f"⚠️ 跳过错误日期：{item.get('update_date')}")

# === 构建完整季度序列，补零 ===
all_quarters = []
for year in range(2020, 2026):
    for q in range(1, 5):
        quarter = f"{year}Q{q}"
        if quarter > "2025Q1":
            break
        all_quarters.append(quarter)

# === 写入CSV文件 ===
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Quarter", "Link Count"])
    for quarter in all_quarters:
        writer.writerow([quarter, quarter_counts.get(quarter, 0)])

print(f"✅ 季度统计结果已保存到 {output_csv_path}")

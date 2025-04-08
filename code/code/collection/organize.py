import json
from datetime import datetime
from collections import defaultdict

# 输入输出路径
input_path = 'LLM_code/dataset/github_links/valid_links.json'
output_path = 'LLM_code/dataset/github_links/valid_links_by_quarter.json'

# 辅助函数：将日期转换为季度标签
def get_quarter(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{quarter}"
    except Exception as e:
        print(f"⚠️ 日期格式错误，跳过：{date_str}")
        return None

# 读取 JSON 数据
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分组
grouped_links = defaultdict(list)
for item in data:
    update_date = item.get("update_date")
    github_link = item.get("github_links")
    quarter = get_quarter(update_date)
    if quarter and github_link:
        grouped_links[quarter].append(github_link)

# 排序后的字典构建
def sort_quarters(quarter_keys):
    def quarter_sort_key(q):
        year, qtr = q.split('Q')
        return int(year) * 10 + int(qtr)
    return sorted(quarter_keys, key=quarter_sort_key)

sorted_grouped_links = {
    quarter: grouped_links[quarter]
    for quarter in sort_quarters(grouped_links.keys())
}

# 写入结果文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_grouped_links, f, indent=4)

print(f"✅ 成功生成按季度排序的链接文件，保存至 {output_path}")

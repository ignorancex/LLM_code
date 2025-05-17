import json
from datetime import datetime
from collections import defaultdict

# 输入输出文件路径
input_path = 'LLM_code/code/github_links/Only_links_set.json'  # 你的原始完整信息文件
output_path = 'LLM_code/code/github_links/Only_links_by_quarter.json'

# 时间范围限制
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 3, 31)

# 辅助函数：将日期转为季度标签
def get_quarter(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if start_date <= dt <= end_date:
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}Q{quarter}"
    except Exception as e:
        print(f"⚠️ 日期解析错误：{date_str}")
    return None

# 加载原始数据
with open(input_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# 构建季度分组字典
grouped = defaultdict(list)

for item in full_data:
    link = item.get("github_links")
    update_date = item.get("update_date")
    quarter = get_quarter(update_date)
    if quarter and link:
        grouped[quarter].append(link)

# 排序季度（从早到晚）
def quarter_sort_key(q):
    year, qtr = q.split('Q')
    return int(year) * 10 + int(qtr)

sorted_grouped = {
    quarter: grouped[quarter]
    for quarter in sorted(grouped.keys(), key=quarter_sort_key)
}

# 写入结果 JSON 文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_grouped, f, indent=4)

print(f"✅ 已生成分季度链接文件，共包含 {len(sorted_grouped)} 个季度，保存为 {output_path}")

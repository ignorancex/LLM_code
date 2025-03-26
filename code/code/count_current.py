import os
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# 加载 JSON 数据
with open('repos.json', 'r', encoding='utf-8') as f:
    repo_data = json.load(f)

# 创建 repo_name -> (link, date) 的映射
repo_map = {
    item["repo"]: (item["github_links"], item["update_date"])
    for item in repo_data
}

# 初始化路径
base_path = "LLM_code/dataset/github_code"
years = ["2020", "2021", "2022", "2023", "2024"]

# 收集所有链接及其 update_date
collected = []

for year in years:
    full_path = os.path.join(base_path, year)
    if not os.path.exists(full_path):
        continue
    for folder in os.listdir(full_path):
        if folder in repo_map:
            link, date = repo_map[folder]
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                collected.append({"link": link, "date": date_obj})
            except Exception as e:
                print(f"❌ 日期解析失败: {date} ({folder})")

# 统计每季度的链接
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 1)
quarterly_stats = defaultdict(list)

current = start_date
while current < end_date:
    next_period = current + relativedelta(months=3)
    period_key = f"{current.date()} to {(next_period - timedelta(days=1)).date()}"
    for item in collected:
        if current <= item["date"] < next_period:
            quarterly_stats[period_key].append({
                "link": item["link"],
                "date": item["date"].strftime("%Y-%m-%d")
            })
    current = next_period

# 保存结果为 JSON 文件
with open("quarterly_github_links.json", "w", encoding="utf-8") as f:
    json.dump(quarterly_stats, f, indent=2, ensure_ascii=False)

# 打印每个季度的链接数量
for period, items in quarterly_stats.items():
    print(f"{period}: {len(items)} links")

import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# 读取 JSON 文件
with open('LLM_code/dataset/github_links/Only_links.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将字符串日期转为 datetime 对象
for item in data:
    item['update_date'] = datetime.strptime(item['update_date'], "%Y-%m-%d")

# 构造时间段：从2020-01-01开始，每3个月一个段，直到2025-01-01
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 4, 1)
periods = []
current = start_date

while current < end_date:
    next_period = current + relativedelta(months=3)
    periods.append((current, next_period))
    current = next_period

# 统计每个时间段的数量
results = []
for start, end in periods:
    count = sum(1 for item in data if start <= item['update_date'] < end)
    period_label = f"{start.date()} to {end.date() - timedelta(days=1)}"
    results.append({"Period": period_label, "Count": count})

# 输出为 CSV 文件
df = pd.DataFrame(results)
df.to_csv("LLM_code/dataset/github_links/period_counts.csv", index=False, encoding='utf-8')

print("统计完成，输出文件为 period_counts.csv")

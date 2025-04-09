import pandas as pd
import os

# 配置参数
input_file = "LLM_code/output_by_quarter/by_project/cumulative_counts/normalized_comments_words_by_project.csv"
output_file = "LLM_code/output_by_quarter/by_project/cumulative_counts/comments_key_words.csv"
x = 0.5  # total 排名前 50%
y = 0.5  # growth_rate 排名前 50%

# 读取数据
df = pd.read_csv(input_file)

# 确保列存在
assert "total" in df.columns, "列 'total' 不存在"
assert "growth_rate" in df.columns, "列 'growth_rate' 不存在"

# 根据 total 排名选前 x 比例
top_x_total = df.sort_values(by="total", ascending=False)
n_total = int(len(top_x_total) * x)
top_total_df = top_x_total.head(n_total)

# 根据 growth_rate 排名选前 y 比例
top_y_growth = df.sort_values(by="growth_rate", ascending=False)
n_growth = int(len(top_y_growth) * y)
top_growth_df = top_y_growth.head(n_growth)

# 并集（合并去重）
combined_df = pd.concat([top_total_df, top_growth_df]).drop_duplicates(subset=["Name"])

# 输出结果
combined_df.to_csv(output_file, index=False)
print(f"筛选后的数据已保存至：{output_file}")

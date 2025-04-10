import pandas as pd

# 读取CSV文件
df = pd.read_csv('LLM_code/output_by_quarter/by_pub/normalization/normalized_nonzero_comment_words_by_pub.csv')

# 需要参与筛选的季度（列）名称，也可以自动从列中筛选，这里直接列举
quarter_cols = [
    '2020Q1','2020Q2','2020Q3','2020Q4',
    '2021Q1','2021Q2','2021Q3','2021Q4',
    '2022Q1','2022Q2','2022Q3','2022Q4',
    '2023Q1','2023Q2','2023Q3','2023Q4',
    '2024Q1','2024Q2','2024Q3','2024Q4',
    '2025Q1'
]

# x 和 y 的比例
x = 0.1
y = 0.5

# 总行数
n = len(df)

# 对所有季度列，逐列计算是否进入前 x%
mask_all_quarters = pd.Series([True]*n, index=df.index)  # 初始化为 True，逐列取交集
for col in quarter_cols:
    # 从大到小排名, rank值从1开始，1表示该列最大的值
    df[col + '_rank'] = df[col].rank(ascending=False, method='first')
    
    # 判断该列是否属于前 x%，即 rank <= x * n
    mask_top_x = df[col + '_rank'] <= x * n
    
    # 累计取 "与" (AND)，确保在每个季度都属于前 x%
    mask_all_quarters = mask_all_quarters & mask_top_x

# 对 growth_rate 列同样进行排名
df['growth_rate_rank'] = df['growth_rate'].rank(ascending=False, method='first')
mask_growth_rate_top_y = df['growth_rate_rank'] <= y * n

# 最后综合筛选：既在所有季度前 x%，又在 growth_rate 前 y%
final_mask = mask_all_quarters & mask_growth_rate_top_y
filtered_df = df[final_mask].copy()

# 想要输出到新的CSV文件，可以去掉 rank 相关的临时列
drop_cols = [col + '_rank' for col in quarter_cols] + ['growth_rate_rank']
filtered_df.drop(columns=drop_cols, inplace=True)

# 将过滤结果写出到 filtered_data.csv
filtered_df.to_csv('key_words.csv', index=False)

print("筛选完成")

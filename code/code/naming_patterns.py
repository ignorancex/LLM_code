import pandas as pd
import re

# 定义五种命名方式的正则表达式
naming_patterns = {
    "camelCase": r'^[a-z]+([A-Z][a-z]*)*$',  # 小写字母开头，后面可以跟一个或多个大写字母开头的单词
    "snake_case": r'^[a-z]+(_[a-z]+)*$',      # 小写字母，单词间用下划线分隔
    "PascalCase": r'^[A-Z][a-z]+([A-Z][a-z]*)*$',  # 大写字母开头，后面可以跟一个或多个大写字母开头的单词
    "UPPER_SNAKE_CASE": r'^[A-Z]+(_[A-Z]+)*$',  # 全大写字母，单词间用下划线分隔
    "Other": r'.*'  # 其他模式
}

# 检查名称是否符合特定的命名模式
def get_naming_pattern(name):
    name = str(name)
    for pattern, regex in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return "Other"

# 读取输入的CSV文件
method = "by_file"
target = "functions"
input_file = f'code/analyze/{method}/data_byfile_{target}.csv'  # 输入文件路径
output_file = f'code/analyze/{method}/{target}_naming_patterns.csv'  # 输出文件路径

df = pd.read_csv(input_file)

# 初始化一个字典来存储每种命名方式在每年中的数量
naming_counts = {year: {pattern: 0 for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}

# 统计每年各个命名方式的数量
for index, row in df.iterrows():
    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
        name = row['Name']
        pattern = get_naming_pattern(name)
        naming_counts[year][pattern] += row[year]

# 计算每年每种命名方式的占比
naming_ratios = {year: {pattern: naming_counts[year][pattern] / row['total'] for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}

# 将占比存储到DataFrame中并输出为CSV文件
output_df = pd.DataFrame(naming_ratios)
output_df.to_csv(output_file)

print(f"Output written to {output_file}")

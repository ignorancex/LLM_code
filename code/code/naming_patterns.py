import pandas as pd
import re

# 定义五种命名方式的正则表达式
naming_patterns = {
    "camelCase": r'^[a-z]+(?:[A-Z][a-z]*)*$',  # 小写字母开头，后面可以跟一个或多个大写字母开头的单词
    "snake_case": r'^[a-z]+(?:_[a-z]+)+$',      # 小写字母，单词间用下划线分隔
    "PascalCase": r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',  # 大写字母开头，后面可以跟一个或多个大写字母开头的单词
    "UPPER_SNAKE_CASE": r'^[A-Z]+(?:_[A-Z]+)+$',  # 全大写字母，单词间用下划线分隔
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
method = "by_project_as_one"
target = "variables"
input_file = f'code/analyze/{method}/data_byproject_{target}_asone.csv'  # 输入文件路径
output_file = f'code/analyze/{method}/{target}_naming_patterns.csv'  # 输出文件路径

df = pd.read_csv(input_file)

# 初始化一个字典来存储每种命名方式在每年中的数量
naming_counts = {year: {pattern: 0 for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
name_lengths = {year: 0 for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
name_value_sums = {year: 0 for year in ['2020', '2021', '2022', '2023', '2024', '2025']}

# 统计每年各个命名方式的数量，同时计算name的长度总和及其数值总和
for index, row in df.iterrows():
    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
        name = str(row['Name'])
        pattern = get_naming_pattern(name)
        value = row[year]
        
        # 统计命名方式出现的数量
        naming_counts[year][pattern] += value
        
        # 累加每年name的长度乘以该年对应的数值
        name_lengths[year] += len(name) * value
        
        # 累加每年数值总和
        name_value_sums[year] += value

# 计算每年每种命名方式的总出现次数
naming_ratios = {year: {pattern: naming_counts[year][pattern] for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}

# 计算每年avg_name_length
avg_name_lengths = {year: name_lengths[year] / name_value_sums[year] if name_value_sums[year] > 0 else 0 for year in name_lengths}

# 将命名方式和avg_name_length放入最终的DataFrame
output_data = []
for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
    row = {'year': year}
    row.update(naming_ratios[year])
    row['avg_name_length'] = avg_name_lengths[year]
    output_data.append(row)

output_df = pd.DataFrame(output_data)

# 输出为CSV文件
output_df.to_csv(output_file, index=False)

print(f"Output written to {output_file}")

import pandas as pd
import re

# 定义五种命名方式的正则表达式
naming_patterns = {
    "single_letter": r'^[a-zA-Z]$', # 单字母变量,x
    "lowercase": r'^[a-z]+$',  # 仅小写字母，不含下划线,countnum
    "UPPERCASE": r'^[A-Z]+$',   # 仅大写字母，不含下划线,COUNTNUM
    "camelCase": r'^[a-z]+(?:[A-Z][a-z]*)*$',  # 小写字母开头，后面可以跟一个或多个大写字母开头的单词,countNum
    "snake_case": r'^[a-z]+(?:_[a-z]+)+$',      # 小写字母，单词间用下划线分隔,count_num
    "PascalCase": r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',  # 大写字母开头，后面可以跟一个或多个大写字母开头的单词,CountNum
    "UPPER_SNAKE_CASE": r'^[A-Z]+(?:_[A-Z]+)+$',  # 全大写字母，单词间用下划线分隔,COUNT_NUM
    "endsWithDigits": r'^[A-Za-z_]+[0-9]+$', # 数字结尾的变量名
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
# input_file = f'LLM_code/code/analyze/{method}/data_byfile_{target}.csv'  # 输入文件路径
# output_file = f'LLM_code/code/analyze/{method}/{target}_naming_pattern.csv'  # 输出文件路径
# ratio_output_file = f'LLM_code/code/analyze/{method}/{target}_naming_ratios.csv'

input_file = 'LLM_code/code/analyze/by_file/data_byfile_file_name_frequency.csv'
output_file = 'LLM_code/code/analyze/by_file/file_naming_pattern.csv'
ratio_output_file = 'LLM_code/code/analyze/by_file/file_naming_ratio.csv'

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


# 生成比例数据并保存为另一个 CSV 文件
ratio_data = []
for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
    total = name_value_sums[year]
    if total == 0:
        continue
    ratio_row = {'year': year}
    for pattern in naming_patterns:
        ratio = naming_counts[year][pattern] / total
        ratio_row[pattern] = round(ratio, 4)  # 保留四位小数
    ratio_data.append(ratio_row)

ratio_df = pd.DataFrame(ratio_data)

# 保存比例文件
ratio_df.to_csv(ratio_output_file, index=False)
print(f"Ratio output written to {ratio_output_file}")
import pandas as pd
import re
import os
from tqdm import tqdm

# 定义命名方式的正则表达式
naming_patterns = {
    "single_letter": r'^[a-zA-Z]$',
    "lowercase": r'^[a-z]+$',
    "UPPERCASE": r'^[A-Z]+$',
    "camelCase": r'^[a-z]+(?:[A-Z][a-z]*)*$',
    "snake_case": r'^[a-z]+(?:_[a-z]+)+$',
    "PascalCase": r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$',
    "UPPER_SNAKE_CASE": r'^[A-Z]+(?:_[A-Z]+)+$',
    "endsWithDigits": r'^[A-Za-z_]+[0-9]+$',
    "Other": r'.*'
}

def get_naming_pattern(name):
    name = str(name)
    for pattern, regex in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return "Other"

# 通用配置
method = "by_mod"
targets = ["functions", "variables", "file_name"]

for target in tqdm(targets, desc="处理多个命名文件"):
    input_file = f'LLM_code/output_by_quarter/{method}/normalization/normalized_{target}_{method}.csv'
    output_file = f'LLM_code/output_by_quarter/{method}/naming_patterns_{target}.csv'

    if not os.path.exists(input_file):
        print(f"未找到输入文件：{input_file}，跳过。")
        continue

    df = pd.read_csv(input_file)

    # 获取所有季度列（排除 Name 和 total）
    quarter_cols = [col for col in df.columns if col not in ["Name", "total", "growth_rate"]]
    quarter_cols = sorted(quarter_cols)  # 按时间顺序排序

    # 初始化统计结构
    naming_counts = {q: {p: 0 for p in naming_patterns} for q in quarter_cols}
    name_lengths = {q: 0 for q in quarter_cols}
    name_value_sums = {q: 0 for q in quarter_cols}

    # 遍历变量名进行统计
    for _, row in df.iterrows():
        name = str(row['Name'])
        pattern = get_naming_pattern(name)

        for quarter in quarter_cols:
            value = row[quarter]
            naming_counts[quarter][pattern] += value
            name_lengths[quarter] += len(name) * value
            name_value_sums[quarter] += value

    # 构造输出 DataFrame
    output_data = []
    for quarter in quarter_cols:
        row = {'quarter': quarter}
        row.update(naming_counts[quarter])
        row['avg_name_length'] = name_lengths[quarter] / name_value_sums[quarter] if name_value_sums[quarter] > 0 else 0
        output_data.append(row)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"{target} 命名模式统计完成：{output_file}")

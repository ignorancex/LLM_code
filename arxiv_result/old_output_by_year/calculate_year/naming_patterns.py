import pandas as pd
import re
naming_patterns = {'single_letter': '^[a-zA-Z]$', 'lowercase': '^[a-z]+$', 'UPPERCASE': '^[A-Z]+$', 'camelCase': '^[a-z]+(?:[A-Z][a-z]*)*$', 'snake_case': '^[a-z]+(?:_[a-z]+)+$', 'PascalCase': '^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', 'UPPER_SNAKE_CASE': '^[A-Z]+(?:_[A-Z]+)+$', 'endsWithDigits': '^[A-Za-z_]+[0-9]+$', 'Other': '.*'}

def get_naming_pattern(name):
    name = str(name)
    for (pattern, regex) in naming_patterns.items():
        if re.match(regex, name):
            return pattern
    return 'Other'
method = 'by_file'
target = 'functions'
input_file = 'LLM_code/code/analyze/by_file/data_byfile_file_name_frequency.csv'
output_file = 'LLM_code/code/analyze/by_file/file_naming_pattern.csv'
ratio_output_file = 'LLM_code/code/analyze/by_file/file_naming_ratio.csv'
df = pd.read_csv(input_file)
naming_counts = {year: {pattern: 0 for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
name_lengths = {year: 0 for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
name_value_sums = {year: 0 for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
for (index, row) in df.iterrows():
    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
        name = str(row['Name'])
        pattern = get_naming_pattern(name)
        value = row[year]
        naming_counts[year][pattern] += value
        name_lengths[year] += len(name) * value
        name_value_sums[year] += value
naming_ratios = {year: {pattern: naming_counts[year][pattern] for pattern in naming_patterns} for year in ['2020', '2021', '2022', '2023', '2024', '2025']}
avg_name_lengths = {year: name_lengths[year] / name_value_sums[year] if name_value_sums[year] > 0 else 0 for year in name_lengths}
output_data = []
for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
    row = {'year': year}
    row.update(naming_ratios[year])
    row['avg_name_length'] = avg_name_lengths[year]
    output_data.append(row)
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file, index=False)
ratio_data = []
for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
    total = name_value_sums[year]
    if total == 0:
        continue
    ratio_row = {'year': year}
    for pattern in naming_patterns:
        ratio = naming_counts[year][pattern] / total
        ratio_row[pattern] = round(ratio, 4)
    ratio_data.append(ratio_row)
ratio_df = pd.DataFrame(ratio_data)
ratio_df.to_csv(ratio_output_file, index=False)
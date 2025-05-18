import os
import pandas as pd
import re

def process_file(base_name):
    base_path = 'output_file'
    years = ['2020', '2021', '2022', '2023', '2024', '2025']
    data = {}
    for year in years:
        file_path = os.path.join(base_path, year, f'{base_name}_{year}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for (_, row) in df.iterrows():
                name = row.iloc[0]
                count = row.iloc[1]
                if name not in data:
                    data[name] = {y: 0 for y in years}
                data[name][year] = count
    merged_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    merged_df.columns = ['Name'] + years
    merged_df['total'] = merged_df[years].sum(axis=1)
    merged_df['rate(2020->2024)'] = merged_df.apply(lambda row: (row['2024'] - row['2020']) / row['2020'] if row['2020'] != 0 else None, axis=1)
    merged_df = merged_df.sort_values(by='total', ascending=False)
    return merged_df
for file in ['functions', 'variables', 'comments_words', 'file_name_frequency']:
    result_df = process_file(file)
    result_df.to_csv(f'data_byfile_{file}.csv', index=False)
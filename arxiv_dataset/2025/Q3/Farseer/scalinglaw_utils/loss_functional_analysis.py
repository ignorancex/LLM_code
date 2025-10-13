
import pandas as pd

def check_value_diff(df: pd.DataFrame, data_recipe: str, value_name: str, N1: int, N2: int) -> pd.DataFrame:
    query_data = df.query(f'data_recipe == "{data_recipe}"')[['D', 'N', f'{value_name}']]
    N1_df = query_data.query(f'N == {N1}').sort_values('D')
    N2_df = query_data.query(f'N == {N2}').sort_values('D')
    merged_df = N1_df.merge(N2_df, on='D', suffixes=('_1', '_2'))
    merged_df['value_diff'] = (merged_df[f'{value_name}_1'] - merged_df[f'{value_name}_2'])
    return merged_df

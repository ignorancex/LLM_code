import pandas as pd
import os
import numpy as np # 用于处理 inf 值

# --- Configuration ---
input_file = "LLM_code/codeforces/simulation/result/case/unique/intersection_vars.csv"

# --- Main Logic ---
if not os.path.exists(input_file):
    print(f"Error: Input file not found at '{input_file}'. Please ensure the intersection CSV is generated.")
    exit()

try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading CSV file '{input_file}': {e}")
    exit()

model_names_in_df = []
# 寻找所有以 '_increase_ratio' 结尾的列名
for col in df.columns:
    if col.endswith('_increase_ratio'):
        model_name = col.replace('_increase_ratio', '')
        model_names_in_df.append(model_name)

if not model_names_in_df:
    print("No '_increase_ratio' columns found in the CSV. Please check the file format.")
    exit()

print(f"Checking variables where 'increase_ratio' is >= 0 for all models: {model_names_in_df}\n")

# 存储符合条件的变量
variables_with_all_positive_increase_ratio = []

# 遍历每一行 (每一个 common_snake_case_var)
for index, row in df.iterrows():
    variable_name = row['variable']
    
    all_ratios_non_negative = True
    for model in model_names_in_df:
        increase_ratio_col = f"{model}_increase_ratio"
        
        # 检查列是否存在，以防万一
        if increase_ratio_col not in row:
            print(f"Warning: Column '{increase_ratio_col}' not found for variable '{variable_name}'. Skipping check for this variable.")
            all_ratios_non_negative = False
            break

        ratio = row[increase_ratio_col]
        
        # 处理可能的 'inf' 或其他非数值情况
        # np.isinf(ratio) 可以判断是否为无穷大
        # ratio >= 0 适用于正数和 inf (因为 inf > 0)
        if not (pd.isna(ratio) or ratio > 0): # 如果是 NaN 或者小于 0
            all_ratios_non_negative = False
            # Optional: print why it failed
            # print(f"  - Variable '{variable_name}' failed: {model}'s ratio is {ratio}")
            break
    
    if all_ratios_non_negative:
        variables_with_all_positive_increase_ratio.append(variable_name)

# 打印结果
if variables_with_all_positive_increase_ratio:
    print("Variables where 'increase_ratio' is >= 0 for all models:")
    for var in variables_with_all_positive_increase_ratio:
        print(f"- {var}")
else:
    print("No variables found where 'increase_ratio' is >= 0 for all models.")

print("\nProcessing complete.")
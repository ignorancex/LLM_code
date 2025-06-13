import pandas as pd
import re
import os
import csv # Explicitly import csv for writer

# --- Configuration ---
csv_files = {
    "DeepSeek": "LLM_code/codeforces/simulation/result/case/unique/DeepSeek_all_variable_frequencies_unique_per_block.csv",
    "Gemma":    "LLM_code/codeforces/simulation/result/case/unique/Gemma_all_variable_frequencies_unique_per_block.csv",
    "Qwen":     "LLM_code/codeforces/simulation/result/case/unique/Qwen_all_variable_frequencies_unique_per_block.csv",
    "Gemini":   "LLM_code/codeforces/simulation/result/case/unique/Gemini_all_variable_frequencies_unique_per_block.csv",
    "GPT":      "LLM_code/codeforces/simulation/result/case/unique/GPT_all_variable_frequencies_unique_per_block.csv",
    "Llama":    "LLM_code/codeforces/simulation/result/case/unique/Llama_all_variable_frequencies_unique_per_block.csv",
}

output_file = "LLM_code/codeforces/simulation/result/case/unique/intersection_vars.csv"
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

# --- Helper Function for snake_case ---
# Regex for snake_case: starts with lowercase, followed by lowercase or underscore,
# not ending with underscore, no consecutive underscores.
# Example matches: my_variable, calculate_sum, total_count
# Example non-matches: MyVariable, _variable, variable_, variable__name, var123
def is_snake_case(variable_name: str) -> bool:
    """Checks if a string conforms to snake_case naming convention."""
    return isinstance(variable_name, str)

# --- Data Processing ---
model_dfs = {}             # Stores DataFrame for each model: {model_name: DataFrame}
model_snake_case_vars = {} # Stores sets of snake_case variables per model: {model_name: Set[str]}

print("Processing files and identifying snake_case variables...")
for model_name, file_path in csv_files.items():
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {model_name} at '{file_path}'. Skipping this model.")
        # Store None for skipped models in model_dfs, and an empty set for intersection
        model_dfs[model_name] = None
        model_snake_case_vars[model_name] = set()
        continue

    try:
        df = pd.read_csv(file_path)
        model_dfs[model_name] = df

        snake_case_vars_in_model = set()
        # Iterate over unique variables to avoid redundant regex checks
        for var in df['variable'].unique():
            if is_snake_case(var):
                snake_case_vars_in_model.add(var)
        model_snake_case_vars[model_name] = snake_case_vars_in_model
        print(f"  - {model_name}: Found {len(snake_case_vars_in_model)} snake_case variables.")
    except Exception as e:
        print(f"Error reading or processing '{file_path}' for {model_name}: {e}. Skipping this model.")
        model_dfs[model_name] = None
        model_snake_case_vars[model_name] = set()

# --- Find Intersection ---
all_snake_case_sets = [s for s in model_snake_case_vars.values() if s is not None]

if not all_snake_case_sets:
    print("No valid snake_case variable sets were found from any processed file. No intersection to compute.")
    exit()

# Initialize common_snake_case_vars with a copy of the first set
common_snake_case_vars = all_snake_case_sets[0].copy()
# Intersect with the rest of the sets
for i in range(1, len(all_snake_case_sets)):
    common_snake_case_vars.intersection_update(all_snake_case_sets[i])

# Sort the common variables for consistent output order
sorted_common_vars = sorted(list(common_snake_case_vars))
print(f"\nFound {len(sorted_common_vars)} common snake_case variables across ALL successfully processed models.")

# --- Prepare Output Data ---
output_rows = []
header = ["variable"]
# Create the header row, ensuring consistent model order
for model_name in csv_files.keys():
    header.extend([
        f"{model_name}_ac_frequency",
        f"{model_name}_ans_frequency",
        f"{model_name}_ref_frequency",
        f"{model_name}_increase_ratio"
    ])
output_rows.append(header)

print("Compiling data for common snake_case variables...")
for var in sorted_common_vars:
    row_data = [var]
    for model_name in csv_files.keys():
        df = model_dfs.get(model_name)
        
        # If the DataFrame for this model was skipped or not loaded successfully
        if df is None:
            # Fill with zeros for models that were skipped
            row_data.extend([0, 0, 0, 0])
            continue
        
        # Find the specific row for the variable in the current model's DataFrame
        var_row_df = df[df['variable'] == var]
        
        if not var_row_df.empty:
            # If the variable exists in this specific model's DataFrame, extract its values
            ac = var_row_df['ac_frequency'].iloc[0]
            ans = var_row_df['ans_frequency'].iloc[0]
            ref = var_row_df['ref_frequency'].iloc[0]
            inc_ratio = var_row_df['increase_ratio'].iloc[0]
            row_data.extend([ac, ans, ref, inc_ratio])
        else:
            # This case should ideally not happen if 'var' is truly in the intersection
            # and 'df' was successfully loaded. But as a safeguard, fill with zeros.
            row_data.extend([0, 0, 0, 0])
    output_rows.append(row_data)

# --- Write to Output CSV ---
try:
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    print(f"\nSuccessfully wrote intersection data to '{output_file}'")
except Exception as e:
    print(f"Error writing output file '{output_file}': {e}")
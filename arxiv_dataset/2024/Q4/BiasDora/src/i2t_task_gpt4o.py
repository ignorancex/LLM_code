import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate significance.")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt4, llama, gpt4o, llava)")
parser.add_argument("--task", type=str, required=True, choices=["t2t", "i2t"], help="Task type (t2t or i2t)")
parser.add_argument("--setting", type=str, required=True, help="Setting number (e.g., 1, 2, 3...)")
args = parser.parse_args()

model = args.model
task = args.task
setting = args.setting

# Define base paths
input_base_path = f"../analysis/{'closed_source' if 'gpt' in model else 'open_source'}/{task}_{model}"
output_base_path = f"../results/{task}_{model}"
fig_base_path = f"../figs/{task}_{model}"

# Define specific paths
input_path = os.path.join(input_base_path, f"setting{setting}_word_cooccurence.csv")
output_csv_path = os.path.join(output_base_path, f"setting{setting}_tfidf_statistics.csv")
output_figure_path = os.path.join(fig_base_path, f"setting{setting}_tfidf_distribution.png")
significant_output_csv_path = os.path.join(output_base_path, f"setting{setting}_significant_tfidf.csv")
highly_significant_output_csv_path = os.path.join(output_base_path, f"setting{setting}_pvalue_significant_tfidf.csv")

# Ensure output directories exist
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(fig_base_path, exist_ok=True)

# Load the CSV data
data = pd.read_csv(input_path)

# **Exclusive Set Filtering for i2t Task and Settings 2, 3, 4, 5**
if task == "i2t" and setting in ["2", "3", "4", "5"]:
    exclude_settings = ["1", "6", "7"]
    exclude_pairs = set()
    
    for exclude_setting in exclude_settings:
        exclude_path = os.path.join(input_base_path, f"setting{exclude_setting}_word_cooccurence.csv")
        if os.path.exists(exclude_path):
            exclude_data = pd.read_csv(exclude_path)
            exclude_pairs.update(zip(exclude_data['target'], exclude_data['word']))

    # Filter out pairs from the current setting data
    data_filtered = data[~data[['target', 'word']].apply(tuple, axis=1).isin(exclude_pairs)]
else:
    data_filtered = data

# Continue with the analysis on the filtered data
# Extract TF-IDF scores
tfidf_scores = data_filtered['tfidf_score'].tolist()

# Convert to a DataFrame for analysis
tfidf_df = pd.DataFrame(tfidf_scores, columns=['TF-IDF Scores'])

# Describe the distribution of TF-IDF scores
distribution = tfidf_df.describe()

# Save the statistics to a CSV file
distribution.to_csv(output_csv_path)

# Prepare a histogram of the TF-IDF scores
plt.figure(figsize=(10, 6))
plt.hist(tfidf_scores, bins=500, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f'Distribution of TF-IDF Scores for {task.upper()} {model.upper()} Setting {setting} with Mean and Standard Deviation')
plt.xlabel('TF-IDF Score')
plt.ylabel('Frequency')

# Add lines for mean and standard deviations
mean = tfidf_df['TF-IDF Scores'].mean()
std_dev = tfidf_df['TF-IDF Scores'].std()
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean + std_dev:.2f}')
plt.axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean - std_dev:.2f}')

# Adjust x-axis limit to include data from min_score to max_score and show std dev lines with minimal buffer
min_score = min(tfidf_scores)
max_score = max(tfidf_scores)
left_limit = max(min_score, mean - 3 * std_dev)
right_limit = min(max_score, mean + 3 * std_dev)
plt.xlim(left_limit, right_limit)

plt.grid(True)
plt.legend()
plt.savefig(output_figure_path)

print(f'Statistical analysis saved to {output_csv_path}')
print(f'Figure saved to {output_figure_path}')

# Calculate the frequency of each (target, word, tfidf_score) combination
data_filtered['frequency'] = data_filtered.groupby(['target', 'word', 'tfidf_score'])['tfidf_score'].transform('count')

# Drop duplicate rows based on 'target', 'word', and 'tfidf_score'
data_unique = data_filtered.drop_duplicates(subset=['target', 'word', 'tfidf_score'])

# Group by 'target' and sort by 'tfidf_score' in descending order within each group
sorted_data = data_unique.groupby('target', sort=False).apply(lambda x: x.sort_values(by='tfidf_score', ascending=False)).reset_index(drop=True)

# Save the sorted significant items to a CSV file
sorted_data.to_csv(significant_output_csv_path, index=False)

print(f'Significant items saved to {significant_output_csv_path}')

# Calculate the mean and standard deviation for the p-value calculation
mean_score = sorted_data['tfidf_score'].mean()
std_score = sorted_data['tfidf_score'].std()

# Calculate p-values assuming a normal distribution
sorted_data['p_value'] = norm.sf(sorted_data['tfidf_score'], loc=mean_score, scale=std_score)

# Define a p-value threshold for significance
p_value_threshold = 0.05  # You can adjust this threshold based on your needs

# Filter for highly significant items
highly_significant_data = sorted_data[sorted_data['p_value'] < p_value_threshold]

# Save the highly significant items to a new CSV file
highly_significant_data.to_csv(highly_significant_output_csv_path, index=False)

print(f'Highly significant items saved to {highly_significant_output_csv_path}')

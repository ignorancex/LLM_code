import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate significance.")
parser.add_argument("--tasks", type=str, nargs='+', required=True, choices=["t2t", "i2t"], help="Task types (t2t or i2t)")
parser.add_argument("--models", type=str, nargs='+', required=True, help="Model names (e.g., gpt4, llama, gpt4o, llava)")
args = parser.parse_args()

tasks = args.tasks
models = args.models

# Setup figure for subplots
fig, axs = plt.subplots(2, 4, figsize=(24, 12))  # 2 rows, 4 columns
fig.tight_layout(pad=5.0)
# fig.suptitle('', fontsize=16)

plot_index = 0  # To keep track of the subplot index

# Iterate over each task and model
for task in tasks:
    for model in models:
        print(f"Processing task: {task}, model: {model}")

        # Define base paths
        input_base_path = f"../analysis/{'closed_source' if 'gpt' in model else 'open_source'}/{task}_{model}"
        output_base_path = f"../src_vis/tfidf/{task}_{model}"
        fig_base_path = f"../figs/tfidf"

        # Ensure output directories exist
        os.makedirs(output_base_path, exist_ok=True)
        os.makedirs(fig_base_path, exist_ok=True)

        # Initialize an empty DataFrame to hold combined data
        combined_data = pd.DataFrame()

        # Iterate over settings 1 to 5 and combine data
        for setting in range(1, 6):
            # Define specific input path
            input_path = os.path.join(input_base_path, f"setting{setting}_word_cooccurence.csv")

            # Load the CSV data if it exists
            if os.path.exists(input_path):
                data = pd.read_csv(input_path)
                combined_data = pd.concat([combined_data, data], ignore_index=True)
            else:
                print(f"Input file {input_path} does not exist, skipping setting {setting}.")

        # Ensure combined data is not empty
        if combined_data.empty:
            print("No data to process. Skipping this model and task.")
            continue

        # Calculate the frequency of each (target, word, tfidf_score) combination
        combined_data['frequency'] = combined_data.groupby(['target', 'word', 'tfidf_score'])['tfidf_score'].transform('count')

        # Drop duplicate rows based on 'target', 'word', and 'tfidf_score'
        data_unique = combined_data.drop_duplicates(subset=['target', 'word', 'tfidf_score'])

        # Group by 'target' and sort by 'tfidf_score' in descending order within each group
        sorted_data = data_unique.groupby('target', sort=False).apply(lambda x: x.sort_values(by='tfidf_score', ascending=False)).reset_index(drop=True)

        # Save the sorted significant items to a CSV file
        significant_output_csv_path = os.path.join(output_base_path, "combined_significant_tfidf.csv")
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
        highly_significant_output_csv_path = os.path.join(output_base_path, "combined_pvalue_significant_tfidf.csv")
        highly_significant_data.to_csv(highly_significant_output_csv_path, index=False)

        print(f'Highly significant items saved to {highly_significant_output_csv_path}')

        # Prepare a histogram of the significant TF-IDF scores
        significant_tfidf_scores = sorted_data['tfidf_score'].tolist()

        # Plot significant TF-IDF scores
        axs[0, plot_index].hist([score for score in significant_tfidf_scores if score <= 250], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axs[0, plot_index].set_title(f'Significant TF-IDF Scores\n{task.upper()} {model.upper()}')
        axs[0, plot_index].set_xlabel('TF-IDF Score')
        axs[0, plot_index].set_ylabel('Frequency')
        # Modify this part to set the xlim to 200 and adjust ticks
        axs[0, plot_index].set_xlim(0, 200)  # Set x-axis limit to 200
        axs[0, plot_index].set_xticks(range(0, 201, 20))  # Set ticks from 0 to 200 with an interval of 2


        mean = pd.Series(significant_tfidf_scores).mean()
        std_dev = pd.Series(significant_tfidf_scores).std()
        axs[0, plot_index].axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
        axs[0, plot_index].axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean + std_dev:.2f}')
        axs[0, plot_index].axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean - std_dev:.2f}')

        # Plot highly significant TF-IDF scores
        highly_significant_tfidf_scores = highly_significant_data['tfidf_score'].tolist()

        axs[1, plot_index].hist([score for score in highly_significant_tfidf_scores if score <= 250], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axs[1, plot_index].set_title(f'Highly Significant TF-IDF Scores\n{task.upper()} {model.upper()}')
        axs[1, plot_index].set_xlabel('TF-IDF Score')
        axs[1, plot_index].set_ylabel('Frequency')
        axs[1, plot_index].set_xlim(0, 200)  # Set x-axis limit to 200
        axs[1, plot_index].set_xticks(range(0, 201, 20))  # Set ticks from 0 to 200 with an interval of 20

        mean_significant = pd.Series(highly_significant_tfidf_scores).mean()
        std_dev_significant = pd.Series(highly_significant_tfidf_scores).std()
        axs[1, plot_index].axvline(mean_significant, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_significant:.2f}')
        axs[1, plot_index].axvline(mean_significant + std_dev_significant, color='green', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean_significant + std_dev_significant:.2f}')
        axs[1, plot_index].axvline(mean_significant - std_dev_significant, color='green', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean_significant - std_dev_significant:.2f}')

        axs[0, plot_index].legend()
        axs[1, plot_index].legend()

        plot_index += 1  # Move to the next subplot

# Save the combined plot
combined_figure_path = os.path.join(fig_base_path, "combined_tfidf_distributions.png")
plt.savefig(combined_figure_path)
plt.close()

print(f'Combined figure saved to {combined_figure_path}')

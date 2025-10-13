import json

import pandas as pd
import matplotlib.pyplot as plt
import os

# File path
_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task/averitec_shared_task_eval_scores.xlsx"
_PATH_DIRECTORY = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task/plots"

# 1. Load Excel file into dataframe
df = pd.read_excel(_PATH)

# 2.a Plot and save distribution of values in various columns as histograms
columns_to_plot = ['semantic_coverage', 'coherence', 'redundancy', 'consistency', 'relevance']

for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=30, edgecolor='black')
    plt.title(f'Distribution of {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(_PATH_DIRECTORY, f'distribution_{column}.png'))  # Saving each plot
    plt.show()


# 2.b Create a summary of scores (count of 1, 2, ..., 5 for each category) and percentages
summary_df = pd.DataFrame()

for column in columns_to_plot:
    score_counts = df[column].value_counts().sort_index()  # Count how many times each score appears
    score_percentages = (score_counts / score_counts.sum()) * 100  # Calculate percentages

    summary_df[f'{column}_count'] = score_counts
    summary_df[f'{column}_percentage'] = score_percentages

# 2.c Save the summary as an Excel sheet
summary_output_path = os.path.join(_PATH_DIRECTORY, 'scores_summary_counts_percentages.xlsx')
summary_df.to_excel(summary_output_path, index=True)

print(f'Summary of score counts and percentages saved to: {summary_output_path}')

# 3. Column 'predicted evidence' contains text, split by space and calculate length
df['text_length'] = df['predicted evidence'].apply(lambda x: len(str(x).split()))

# Plot and save resulting text length as a histogram
plt.figure(figsize=(8, 6))
plt.hist(df['text_length'], bins=30, edgecolor='black')
plt.title('Distribution of Text Length in Predicted Evidence')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.savefig(os.path.join(_PATH_DIRECTORY, 'distribution_text_length.png'))
plt.show()

# 4. Plot and save the correlation between text length and 'hmeteor' values
plt.figure(figsize=(8, 6))
plt.scatter(df['text_length'], df['hmeteor'])
plt.title('Text Length vs. Hmeteor')
plt.xlabel('Text Length')
plt.ylabel('Hmeteor')
plt.savefig(os.path.join(_PATH_DIRECTORY, 'text_length_vs_hmeteor.png'))
plt.show()

# 5. Normalize labels in 'label_majority' and 'pred_label'
df['label_majority_normalized'] = df['label_majority'].str.strip().str.lower().replace({
    'conflicting/cherry-picking': 'conflicting evidence/cherrypicking',
    'conflicting evidence/cherrypicking': 'conflicting evidence/cherrypicking'
})

df['pred_label_normalized'] = df['pred_label'].str.strip().str.lower().replace({
    'conflicting/cherry-picking': 'conflicting evidence/cherrypicking',
    'conflicting evidence/cherrypicking': 'conflicting evidence/cherrypicking'
})

# 6. Compare how often 'label_majority' and 'pred_label' are the same
df['match'] = df['label_majority_normalized'] == df['pred_label_normalized']
overlap_count = df['match'].sum()
total_count = len(df)
overlap_percentage = (overlap_count / total_count) * 100

print(f'Number of matches: {overlap_count}')
print(f'Percentage of matches: {overlap_percentage:.2f}%')

# 7. Breakdown of how labels in 'label_majority' map to 'pred_label'
label_mapping = pd.crosstab(df['label_majority_normalized'], df['pred_label_normalized'], margins=True)

print("\nLabel Mapping Breakdown:")
print(label_mapping)

# 8. Save the crosstab table as an Excel file
crosstab_output_path = os.path.join(_PATH_DIRECTORY, 'label_mapping_crosstab.xlsx')
label_mapping.to_excel(crosstab_output_path)

# 9. Breakdown of how labels in 'label_majority' map to 'pred_label' (percentages)
label_mapping_percent = pd.crosstab(df['label_majority_normalized'], df['pred_label_normalized'], normalize='index', margins=True) * 100

# 10. Save the crosstab table (percentages) as an Excel file
crosstab_percent_output_path = os.path.join(_PATH_DIRECTORY, 'label_mapping_crosstab_percent.xlsx')
label_mapping_percent.to_excel(crosstab_percent_output_path)
print(f'Crosstab table (percentages) saved to: {crosstab_percent_output_path}')


# 11. Eval statistics per team

# Path to the source file
_PATH_SOURCE_FILE = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task/averitec_shared_task_eval_scores_source.xlsx"

# Load the source data
data_source = pd.read_excel(_PATH_SOURCE_FILE, header=0)

# Print an overview table of how often each entry in the "Team" column occurs
team_counts = data_source['source'].value_counts()
print("Team counts:\n", team_counts)

# Calculate the average score for "semantic_coverage" per team
team_avg_semantic_coverage = data_source.groupby('source')['semantic_coverage'].mean()

# Rank teams according to their average semantic coverage score
team_ranking = team_avg_semantic_coverage.sort_values(ascending=False)

# Display the results
print("\nTeam average semantic coverage score:\n", team_avg_semantic_coverage)
print("\nTeams ranked by semantic coverage score:\n", team_ranking)

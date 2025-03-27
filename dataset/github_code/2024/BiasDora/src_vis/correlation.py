import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_bias_mapping(json_file):
    with open(json_file, 'r') as f:
        bias_mapping = json.load(f)
    return bias_mapping

def map_bias_dimension(target, bias_mapping):
    for dimension, targets in bias_mapping.items():
        if target in targets:
            return dimension
    return 'Unknown'  # Default if no match is found

setting_titles = ['Singular', 'Plural', 'Adjective', 'Noun', 'Verb']
standard_order = ['age', 'disability', 'gender', 'nationality', 'physical-appearance', 'race-color', 'religion', 'sexual-orientation', 'socioeconomic']

def calculate_correlation_matrix(df1, df2, bias_column):
    # Filter for negative sentiments
    df1_negative = df1[df1['concatenated_sentence_sentiment'] == 'NEGATIVE']
    df2_negative = df2[df2['concatenated_sentence_sentiment'] == 'NEGATIVE']

    # Group by bias dimension and count the occurrences
    counts1 = df1_negative[bias_column].value_counts().reindex(standard_order, fill_value=0)
    counts2 = df2_negative[bias_column].value_counts().reindex(standard_order, fill_value=0)

    # Convert to DataFrame for correlation analysis
    df1_counts = pd.DataFrame(counts1).T
    df2_counts = pd.DataFrame(counts2).T

    # Rename columns to match bias dimensions
    df1_counts.columns = standard_order
    df2_counts.columns = standard_order

    # Combine the two dataframes
    combined_df = pd.concat([df1_counts, df2_counts], keys=['t2t_gpt4', 't2t_llama'])

    # Compute correlation matrix
    correlation_matrix = combined_df.corr()
    return correlation_matrix

def generate_figure(mode, settings, save_dir, base_paths):
    os.makedirs(save_dir, exist_ok=True)

    sns.set_context("paper", font_scale=1.5)
    fig, axes = plt.subplots(1, len(settings), figsize=(50, 8), sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for j, setting in enumerate(settings):
        ax = axes[j]
        file_path_gpt4 = os.path.join(base_paths['t2t_gpt4'], f'{setting}_pvalue_significant_tfidf_sentiments.csv')
        file_path_llama = os.path.join(base_paths['t2t_llama'], f'{setting}_pvalue_significant_tfidf_sentiments.csv')

        if os.path.exists(file_path_gpt4) and os.path.exists(file_path_llama):
            df_gpt4 = pd.read_csv(file_path_gpt4)
            df_llama = pd.read_csv(file_path_llama)

            df_gpt4['bias_dimension'] = df_gpt4['id'].str.extract(r'([^_]+)_[\d]+')[0]
            df_llama['bias_dimension'] = df_llama['id'].str.extract(r'([^_]+)_[\d]+')[0]

            df_gpt4['bias_dimension'] = pd.Categorical(df_gpt4['bias_dimension'], categories=standard_order, ordered=True)
            df_llama['bias_dimension'] = pd.Categorical(df_llama['bias_dimension'], categories=standard_order, ordered=True)

            # Calculate correlation matrix
            correlation_matrix = calculate_correlation_matrix(df_gpt4, df_llama, 'bias_dimension')

            # Plot correlation matrix
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Pastel1', ax=ax, vmin=-1, vmax=1,
                        xticklabels=standard_order, yticklabels=standard_order, cbar=True, cbar_kws={'shrink': 0.5})

            ax.set_title(setting_titles[j], fontsize=20, fontweight='bold')
            ax.set_xticklabels(standard_order, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(standard_order, rotation=0, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_file = os.path.join(save_dir, f'correlation_analysis_{mode}.png')
    plt.savefig(combined_file, dpi=300)
    plt.close()

    print(f"Correlation analysis visualization saved successfully in: {combined_file}")

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Generate correlation analysis figures for t2t and i2t models.")
    parser.add_argument('mode', choices=['t2t', 'i2t'], help="Mode to generate the figure for ('t2t' or 'i2t').")

    args = parser.parse_args()

    if args.mode == 't2t':
        base_paths = {
            't2t_gpt4': '../analysis/closed_source/t2t_gpt4',
            't2t_llama': '../analysis/open_source/t2t_llama'
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        save_dir = '../figs/visualizations/t2t'
    elif args.mode == 'i2t':
        base_paths = {
            'i2t_gpt4o': '../analysis/closed_source/i2t_gpt4o',
            'i2t_llava': '../analysis/open_source/i2t_llava'
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        save_dir = '../figs/visualizations/i2t'

    generate_figure(args.mode, settings, save_dir, base_paths)

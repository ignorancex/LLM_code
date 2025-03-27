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

def normalize_data(df, bias_column, sentiment_column):
    print("Available columns in DataFrame:", df.columns)  # Debug statement
    # Group by bias dimension and sentiment, then count the occurrences
    counts = df.groupby([bias_column, sentiment_column]).size().unstack(fill_value=0)
    
    # Normalize counts by dividing by the total counts for each bias dimension
    normalized_counts = counts.div(counts.sum(axis=1), axis=0)
    
    # Flatten the DataFrame to a long format
    normalized_df = normalized_counts.reset_index().melt(id_vars=[bias_column], value_name='percentage', var_name=sentiment_column)

    return normalized_df


def generate_figure(mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping):
    os.makedirs(save_dir, exist_ok=True)

    sns.set_context("paper", font_scale=2.0)
    fig, axes = plt.subplots(len(models), len(settings), figsize=(25, 10), sharey=False)
    plt.subplots_adjust(hspace=10)

    setting_titles = ['Singular', 'Plural', 'Adjective', 'Noun', 'Verb']
    # setting_titles = ['Objective', 'Subjective', 'Stereotypical', 'Implicit', 'Lexical']
    standard_order = ['age', 'disability', 'gender', 'nationality', 'physical-appearance', 'race-color', 'religion', 'sexual-orientation', 'socioeconomic']
    original_to_new_labels = {
        'age': 'AG',
        'disability': 'DA',
        'gender': 'GE',
        'sexual-orientation': 'SO',
        'physical-appearance': 'PA',
        'socioeconomic': 'SE',
        'nationality': 'NT',
        'race-color': 'RC',
        'religion': 'RE'
    }
    new_labels = [original_to_new_labels.get(label, label) for label in standard_order]

    row_titles = ['GPT-4o', 'Llama-3-8B']

    for i, model in enumerate(models):
        for j, setting in enumerate(settings):
            ax = axes[i, j]
            file_path = os.path.join(base_paths[model], f'{setting}_pvalue_significant_tfidf_sentiments.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if mode == 't2t':
                    df['bias_dimension'] = df['id'].str.extract(r'([^_]+)_[\d]+')[0]
                elif mode == 'i2t':
                    df['bias_dimension'] = df['target'].apply(lambda x: map_bias_dimension(x, bias_mapping))

                df['bias_dimension'] = pd.Categorical(df['bias_dimension'], categories=standard_order, ordered=True)

                # Normalize the data
                normalized_df = normalize_data(df, 'bias_dimension', sentiment_column)

                sentiment_order = ['POSITIVE', 'NEGATIVE'] #['NEGATIVE']
                sns.barplot(data=normalized_df, x='bias_dimension', y='percentage', hue=sentiment_column, hue_order=sentiment_order,
                            palette={'NEGATIVE': '#ff8787', 'POSITIVE': '#3bc9db'}, ax=ax, order=standard_order)

                ax.set_xticklabels(new_labels, rotation=45, ha='right')
                ax.set_title(setting_titles[j], fontsize=20, fontweight='bold')

                # Set y-axis limit to 0-1 for normalization, but display as percentages
                ax.set_ylim(0, 1)
                ax.set_yticks([i/100 for i in range(0, 101, 20)])
                ax.set_yticklabels([f'{i}' for i in range(0, 101, 20)])

                # Add y-axis label "Percentage" only to the first subplot in each row
                if j == 0:
                    ax.set_ylabel('Percentage')
                else:
                    ax.set_ylabel('')



                if mode == 'i2t':
                    if setting in ['setting1', 'setting2', 'setting3']:
                        ax.set_ylim(0, 1)
                        ax.set_yticks([i/10 for i in range(11)])
                    elif setting in ['setting4', 'setting5']:
                        ax.set_ylim(0, 1)
                        ax.set_yticks([i/10 for i in range(11)])
                elif mode == 't2t':
                    ax.set_ylim(0, 1)
                    ax.set_yticks([i/10 for i in range(11)])

                if i == 0 and j == len(settings) - 1:
                    legend = ax.legend(title='', loc='upper right')
                    for text in legend.get_texts():
                        if text.get_text() == 'POSITIVE':
                            text.set_text('positive')
                        elif text.get_text() == 'NEGATIVE':
                            text.set_text('negative')
                else:
                    ax.legend().set_visible(False)

                if j == 0:
                    ax.set_ylabel(row_titles[i], fontsize=20, fontweight='bold', rotation=90, labelpad=80, va='bottom')
                    ax.yaxis.set_label_coords(-0.27, 0.5)  # Adjust the coordinates to place it at the top

    for ax in axes.flatten():
        ax.set_xlabel('')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_file = os.path.join(save_dir, f'sentiment_distribution_{mode}.png')
    plt.savefig(combined_file, dpi=300)
    plt.close()

    print(f"Combined visualization saved successfully in: {combined_file}")

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Generate sentiment distribution figures for t2t and i2t models.")
    parser.add_argument('mode', choices=['t2t', 'i2t'], help="Mode to generate the figure for ('t2t' or 'i2t').")

    args = parser.parse_args()

    if args.mode == 't2t':
        base_paths = {
            't2t_gpt4': '../analysis/closed_source/t2t_gpt4',
            't2t_llama': '../analysis/open_source/t2t_llama'
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        models = ['t2t_gpt4', 't2t_llama']
        sentiment_column = 'concatenated_sentence_sentiment'
        save_dir = '../figs/visualizations/t2t'
        bias_mapping = {}  # No bias mapping needed for t2t mode
    elif args.mode == 'i2t':
        base_paths = {
            'i2t_gpt4o': '../analysis/closed_source/i2t_gpt4o',
            'i2t_llava': '../analysis/open_source/i2t_llava'
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5'] #, 'setting6', 'setting7']
        models = ['i2t_gpt4o', 'i2t_llava']
        sentiment_column = 'word_sentiment'
        save_dir = '../figs/visualizations/i2t'
        bias_mapping = load_bias_mapping('../data/bias_target_mapping.json')

    generate_figure(args.mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping)
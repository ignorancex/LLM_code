import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from math import pi

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
    # Group by bias dimension and sentiment, then count the occurrences
    counts = df.groupby([bias_column, sentiment_column]).size().unstack(fill_value=0)
    # Normalize counts by dividing by the total counts for each bias dimension
    normalized_counts = counts.div(counts.sum(axis=1), axis=0)
    return normalized_counts

def generate_figure(mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping):

    # Define the order of bias dimensions
    standard_order = ['age', 'disability', 'gender', 'nationality', 'physical-appearance', 'race-color', 'religion', 'sexual-orientation', 'socioeconomic']

    # Mapping of settings to custom labels
    custom_labels = {
        't2t': ['Singular', 'Plural', 'Adjective', 'Noun', 'Verb'],
        'i2t': ['Objective', 'Subjective', 'Stereotypical', 'Implicit', 'Lexical']
    }

    # Mapping of original dimensions to new labels
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

    # Generate Set2 colors
    # set2_colors = ["aqua", "lightcoral", "lightgreen", "teal", "olive"]
    set2_colors = sns.color_palette('husl', 5) #n_colors=len(standard_order))


    # Define standard yticks for radial gridlines
    yticks1 = [0.2, 0.3]
    yticks2 = [0.2, 0.3]
    yticklabels1 = [str(y) for y in yticks1]
    yticklabels2 = [str(y) for y in yticks2]

    # Create subplots for t2t models
    if mode == 't2t':
        fig_t2t, (ax1_t2t, ax2_t2t) = plt.subplots(1, 2, figsize=(25, 10), subplot_kw=dict(polar=True))

        # Set yticks and labels for both axes
        ax1_t2t.set_yticks(yticks1)
        ax1_t2t.set_yticklabels(yticklabels1, fontsize=15)
        ax2_t2t.set_yticks(yticks2)
        ax2_t2t.set_yticklabels(yticklabels2, fontsize=15)

        fig_t2t.text(0.1, 0.9, 'GPT-4o', transform=fig_t2t.transFigure, fontsize=30, fontweight='bold', va='center', rotation=90)
        fig_t2t.text(0.55, 0.9, 'Stable Diffusion and LLaVA', transform=fig_t2t.transFigure, fontsize=30, fontweight='bold', va='center', rotation=90)

        legend_handles = []  # To store legend handles for each setting

        for i, (model, ax) in enumerate(zip(models, [ax1_t2t, ax2_t2t])):
            if model in base_paths['t2t']:
                angles = [n / float(len(standard_order)) * 2 * pi for n in range(len(standard_order))]
                angles += angles[:1]  # Complete the loop

                # Modify the existing inner circle gridline
                ax.yaxis.grid(True, linestyle='--', linewidth=4, color='#D3D3D3')

                # Change the color of the outermost circle
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D3D3D3')
                    spine.set_linewidth(4)  # Setting linewidth
                    spine.set_linestyle('--')  # Setting linestyle

                for j, setting in enumerate(settings):
                    file_path = os.path.join(base_paths['t2t'][model], f'{setting}_pvalue_significant_tfidf_sentiments.csv')
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)

                        df['bias_dimension'] = df['id'].str.extract(r'([^_]+)_[\d]+')[0]
                        setting_label = custom_labels['t2t'][j]

                        df['bias_dimension'] = df['bias_dimension'].map(original_to_new_labels)
                        df['bias_dimension'] = pd.Categorical(df['bias_dimension'], categories=list(original_to_new_labels.values()), ordered=True)

                        normalized_counts = normalize_data(df, 'bias_dimension', sentiment_column)

                        stats = normalized_counts.get('NEGATIVE', pd.Series([0]*len(standard_order))).tolist()
                        stats += stats[:1]

                        # Use Set2 colors for plot and fill
                        color_index = j % len(set2_colors)
                        line, = ax.plot(angles, stats, 'o-', markersize=10, linewidth=4, label=f'{setting_label}', color=set2_colors[color_index])
                        ax.fill(angles, stats, alpha=0.25, color=set2_colors[color_index])

                        if i == 0:  # Collect legend handles only once
                            legend_handles.append(line)
                
                ax.set_rlabel_position(0)  # Set radial labels to the center
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(list(original_to_new_labels.values()), fontsize=30)  # Adjust font size for xtick labels
                yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
                # ax.set_yticks(yticks)
                # ax.set_yticks(yticks)
                ax.tick_params(axis='y', labelsize=30)  # Correct method to adjust font size
                ax.grid(True)

        # Place legend outside the subplots
        fig_t2t.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles], loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05), fontsize=30)

        # Save the combined t2t figure
        combined_file_t2t = os.path.join(save_dir, f'sent_t2t.png')
        plt.savefig(combined_file_t2t, dpi=300, bbox_inches='tight')  # Adjusted to include tight layout for saving
        plt.close(fig_t2t)
        print(f"Spider visualization saved successfully in: {combined_file_t2t}")

    elif mode == 'i2t':
        # Create subplots for i2t models
        fig_i2t, (ax1_i2t, ax2_i2t) = plt.subplots(1, 2, figsize=(25, 10), subplot_kw=dict(polar=True))

        # Set yticks and labels for both axes
        ax1_i2t.set_yticks(yticks1)
        ax1_i2t.set_yticklabels(yticklabels1, fontsize=15)
        ax2_i2t.set_yticks(yticks2)
        ax2_i2t.set_yticklabels(yticklabels2, fontsize=15)

        fig_i2t.text(0.0, 0.5, '(a)', transform=fig_i2t.transFigure, fontsize=30, fontweight='bold', va='center', rotation=90)
        fig_i2t.text(0.55, 0.2, '', transform=fig_i2t.transFigure, fontsize=30, fontweight='bold', va='bottom', rotation=90)

        legend_handles = []  # To store legend handles for each setting

        for i, (model, ax) in enumerate(zip(models, [ax1_i2t, ax2_i2t])):
            if model in base_paths['i2t']:
                angles = [n / float(len(standard_order)) * 2 * pi for n in range(len(standard_order))]
                angles += angles[:1]  # Complete the loop

                # Modify the existing inner circle gridline
                ax.yaxis.grid(True, linestyle='--', linewidth=4, color='#D3D3D3')

                # Change the color of the outermost circle
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D3D3D3')
                    spine.set_linewidth(4)  # Setting linewidth
                    spine.set_linestyle('--')  # Setting linestyle

                for j, setting in enumerate(settings):
                    file_path = os.path.join(base_paths['i2t'][model], f'{setting}_pvalue_significant_tfidf_sentiments.csv')
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)

                        df['bias_dimension'] = df['target'].apply(lambda x: map_bias_dimension(x, bias_mapping))
                        setting_label = custom_labels['i2t'][j]

                        df['bias_dimension'] = df['bias_dimension'].map(original_to_new_labels)
                        df['bias_dimension'] = pd.Categorical(df['bias_dimension'], categories=list(original_to_new_labels.values()), ordered=True)

                        normalized_counts = normalize_data(df, 'bias_dimension', sentiment_column)

                        stats = normalized_counts.get('NEGATIVE', pd.Series([0]*len(standard_order))).tolist()
                        stats += stats[:1]

                        # Use Set2 colors for plot and fill
                        color_index = j % len(set2_colors)
                        line, = ax.plot(angles, stats, 'o-', linewidth=4, label=f'{setting_label}', color=set2_colors[color_index])
                        ax.fill(angles, stats, alpha=0.25, color=set2_colors[color_index])

                        if i == 0:  # Collect legend handles only once
                            legend_handles.append(line)

                ax.set_rlabel_position(-90)  # Set radial labels to the center
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(list(original_to_new_labels.values()), fontsize=30)  # Adjust font size for xtick labels
                ax.tick_params(axis='y', labelsize=30)  # Correct method to adjust font size
                ax.grid(True)

        # Place legend outside the subplots
        fig_i2t.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles], loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05), fontsize=30)                                                                                     
        # fig_i2t.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles], loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=20)

        # Save the combined i2t figure
        combined_file_i2t = os.path.join(save_dir, f'sent_i2t.png')
        plt.savefig(combined_file_i2t, dpi=300, bbox_inches='tight')  # Adjusted to include tight layout for saving
        plt.close(fig_i2t)
        print(f"Spider visualization saved successfully in: {combined_file_i2t}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment distribution figures for t2t and i2t models.")
    parser.add_argument('mode', choices=['t2t', 'i2t'], help="Mode to generate the figure for ('t2t' or 'i2t').")

    args = parser.parse_args()

    bias_mapping = {}

    if args.mode == 't2t':
        base_paths = {
            't2t': {
                't2t_gpt4': '../analysis/closed_source/t2t_gpt4',
                't2t_llama': '../analysis/open_source/t2t_llama'
            },
            'i2t': {}
        }
        models = ['t2t_gpt4', 't2t_llama']
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        sentiment_column = 'concatenated_sentence_sentiment'
        save_dir = '../figs/visualizations/t2t'
    elif args.mode == 'i2t':
        base_paths = {
            't2t': {},
            'i2t': {
                'i2t_gpt4o': '../analysis/closed_source/i2t_gpt4o',
                'i2t_llava': '../analysis/open_source/i2t_llava'
            }
        }
        models = ['i2t_gpt4o', 'i2t_llava']
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        sentiment_column = 'word_sentiment'
        save_dir = '../figs/visualizations/i2t'
        bias_mapping = load_bias_mapping('../data/bias_target_mapping.json')

    generate_figure(args.mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping)

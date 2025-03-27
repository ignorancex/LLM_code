import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json
from matplotlib.colors import Normalize
sns.set_context("paper")

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

    # Flatten the DataFrame to a long format
    normalized_df = normalized_counts.reset_index().melt(id_vars=[bias_column], value_name='percentage', var_name=sentiment_column)

    return normalized_df

def generate_combined_heatmap(mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    sns.set_context("paper", font_scale=2.0)
    # Define the standard order and the mapping to new labels
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

    # Define the sentiment mapping for scale assessment
    sentiment_levels = {
        1: 'Not at all',
        2: 'Slightly',
        3: 'Moderately',
        4: 'Highly',
        5: 'Extremely'
    }

    fig, axes = plt.subplots(len(models), len(settings), figsize=(40, 30), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Titles for the columns in the specified order
    # column_titles = ['Singular', 'Plural', 'Adjective', 'Noun', 'Verb']
    column_titles = ['Objective', 'Subjective', 'Stereotypical', 'Implicit', 'Lexical']

    # Titles for each row
    row_titles = ['(c)', '(d)']
    # row_titles = ['DALL-E/GPT4-o', 'StableDiffusion/Llava']

    # Modify the heatmap plotting part
    for i, model in enumerate(models):
        for j, setting in enumerate(settings):
            ax = axes[i, j]

            # Define file path for the current setting and model
            if model in ['t2t_llama', 'i2t_llava']:
                if mode == 't2t':
                    file_path = f'../analysis/closed_source_llama/{setting}_t2t_scale_assessment.csv'
                elif mode == 'i2t':
                    file_path = f'../analysis/closed_source_llama/{setting}_i2t_scale_assessment.csv'
            else:
                file_path = os.path.join(base_paths[model], f'{setting}_scale_assessment.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if mode == 't2t':
                    df['bias_dimension'] = df['id'].str.extract(r'([^_]+)_[\d]+')[0]
                elif mode == 'i2t':
                    df['bias_dimension'] = df['target'].apply(lambda x: map_bias_dimension(x, bias_mapping))

                # Ensure the bias_dimension is a categorical with the specified order
                df['bias_dimension'] = pd.Categorical(df['bias_dimension'], categories=standard_order, ordered=True)

                # Remove rows with non-finite values in the sentiment column
                df = df[df[sentiment_column].notnull()]

                # Replace numerical sentiment levels with descriptive labels
                df[sentiment_column] = df[sentiment_column].astype(int).replace(sentiment_levels)

                # Normalize the data
                normalized_df = normalize_data(df, 'bias_dimension', sentiment_column)

                # Create a pivot table for heatmap
                heatmap_data = normalized_df.pivot(index='bias_dimension', columns=sentiment_column, values='percentage').fillna(0)

                # Reorder the rows based on standard_order
                heatmap_data = heatmap_data.loc[standard_order]
                # Reorder columns to reflect the specified sentiment levels order
                heatmap_data = heatmap_data[['Not at all', 'Slightly', 'Moderately', 'Highly', 'Extremely']]

                # Create a truncated colormap
                def truncate_colormap(cmap, min_val=0.3, max_val=1.0, n=50):
                    new_cmap = mcolors.LinearSegmentedColormap.from_list(
                        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
                        cmap(np.linspace(min_val, max_val, n))
                    )
                    return new_cmap

                # Truncate the GnBu colormap to exclude the darkest shades
                truncated_cmap = truncate_colormap(plt.cm.Spectral, min_val=0.3, max_val=0.9)

                # Create a list of truncated colormaps for each column
                colormaps = [truncated_cmap for _ in range(heatmap_data.shape[1])]

                # Define a color map for each sentiment level
                # colormaps = [plt.cm.GnBu, plt.cm.GnBu, plt.cm.GnBu, plt.cm.GnBu, plt.cm.GnBu]

                # Create a normalized version of the heatmap data for coloring
                normed_heatmap_data = heatmap_data.copy()
                for col in heatmap_data.columns:
                    norm = Normalize(vmin=0, vmax=heatmap_data[col].max())
                    normed_heatmap_data[col] = norm(heatmap_data[col].values)

                # Plot heatmap without annotations
                sns.heatmap(
                    heatmap_data,
                    annot=False,  # Remove automatic annotation
                    fmt='.2f',
                    cbar=False,
                    xticklabels=True,
                    yticklabels=new_labels,
                    ax=ax,
                    linewidths=0,
                    linecolor='black'
                )

                # Apply the colors manually by iterating over each cell
                for k, col in enumerate(heatmap_data.columns):
                    for idx, val in enumerate(heatmap_data[col]):
                        norm_val = normed_heatmap_data[col][idx]
                        rgba_color = colormaps[k](norm_val ** 2)  # Adjust intensity with exponent
                        hex_color = mcolors.to_hex(rgba_color)
                        ax.add_patch(plt.Rectangle((k, idx), 1, 1, fill=True, color=hex_color, edgecolor='black', lw=1.0))
                        
                        # Add text annotation with black font color
                        percentage_value = int(val * 100)  # Convert to integer percentage
                        ax.text(k + 0.5, idx + 0.5, f'{val:.2f}', color='black', ha='center', va='center', size=35)
                        # ax.text(k + 0.5, idx + 0.5, f'{percentage_value}', color='black', ha='center', va='center', size=30)
                # for ticklabel, tickcolor in zip(ax.get_xticklabels(), colormaps):
                #     ticklabel.set_color(tickcolor(0.7))  # Set the color of the tick labels to match the colormap

                # Add inner cell borders
                for k in range(len(heatmap_data.columns)):
                    for idx in range(len(heatmap_data)):
                        # Draw rectangle around each cell to act as a border
                        rect = plt.Rectangle((k, idx), 1, 1, fill=False, edgecolor='black', linewidth=3.0)
                        ax.add_patch(rect)

                outer_border = plt.Rectangle((0, 0), len(heatmap_data.columns), len(heatmap_data), 
                             fill=False, edgecolor='black', linewidth=8)
                ax.add_patch(outer_border)

                ax.set_xlabel('')  # Remove the xlabel
                ax.set_ylabel('')  # Remove the ylabel

                # Set title and labels
                if i == 0:
                    ax.set_title(column_titles[j], fontsize=60, fontweight='bold')
                else:
                    ax.set_title('')

                if j == 0:
                    ax.set_ylabel(row_titles[i], fontsize=50, fontweight='bold', rotation=0, labelpad=80, va='bottom')
                    ax.yaxis.set_label_coords(-0.7, 0.94)  # Adjust the coordinates to place it at the top

                # Set fontsize and rotation for ticklabels
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=50, rotation=90, ha='center')
                ax.set_yticklabels(new_labels, fontsize=50, rotation=0, ha='right')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the combined figure
    combined_file = os.path.join(save_dir, f'combined_heatmap_{mode}.png')
    plt.savefig(combined_file, dpi=300)
    plt.close()

    print(f"Combined heatmap saved successfully in: {combined_file}")

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Generate sentiment distribution figures for t2t and i2t models.")
    parser.add_argument('mode', choices=['t2t', 'i2t'], help="Mode to generate the figure for ('t2t' or 'i2t').")

    args = parser.parse_args()

    if args.mode == 't2t':
        base_paths = {
            't2t_gpt4': '../analysis/closed_source/t2t_gpt4',
            't2t_llama': '../analysis/closed_source_llama'  # Updated for t2t_llama
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        models = ['t2t_gpt4', 't2t_llama']
        sentiment_column = 'response'
        save_dir = '../figs/visualizations/t2t'
        bias_mapping = {}  # No bias mapping needed for t2t mode
    elif args.mode == 'i2t':
        base_paths = {
            'i2t_gpt4o': '../analysis/closed_source/i2t_gpt4o',
            'i2t_llava': '../analysis/closed_source_llama'  # Updated for i2t_llava
        }
        settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5']
        models = ['i2t_gpt4o', 'i2t_llava']
        sentiment_column = 'response'
        save_dir = '../figs/visualizations/i2t'
        bias_mapping = load_bias_mapping('../data/bias_target_mapping.json')

    generate_combined_heatmap(args.mode, models, settings, sentiment_column, save_dir, base_paths, bias_mapping)

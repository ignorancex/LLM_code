import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Set save folder
SAVE_DIR_NAME = "python_paper_plots"

# Configurations
FACTUAL_COLOR = "#005CAB"
FACTUAL_CMAP = sns.diverging_palette(10, 250, as_cmap=True)
COUNTERFACTUAL_COLOR = "#E31B23"
COUNTERFACTUAL_CMAP = sns.diverging_palette(300, 10, as_cmap=True)

AXIS_TITLE_SIZE = 20
AXIS_TEXT_SIZE = 15

#########################################
#### Heatmap: Layer-Head Combination ####
#########################################

# Create a heatmap function
def create_heatmap(data, relevant_position, midpoint=0):
    plt.figure(figsize=(10, 8))

    # Create the pivot table for all data
    data_pivot = data.pivot(index="y_label", columns="mapped_position", values="value")

    # Create masks for Target and Other
    mask_target = ~(data.pivot(index="y_label", columns="mapped_position", values="color") == "Target")
    mask_other = ~(data.pivot(index="y_label", columns="mapped_position", values="color") == "Other")

    # Plot both heatmaps with their respective masks
    ax1 = sns.heatmap(data_pivot,
                mask=mask_target,
                cmap=FACTUAL_CMAP,
                linewidths=0.5,
                center=midpoint,
                cbar_kws={"label": "Attention Score Factual", "pad": 0.1},
                xticklabels=relevant_position,
                yticklabels=True)

    ax2 = sns.heatmap(data_pivot,
                mask=mask_other,
                cmap=COUNTERFACTUAL_CMAP,
                linewidths=0.5,
                center=midpoint,
                cbar_kws={"label": "Attention Score Counterfactual", "pad": 0.1},
                xticklabels=relevant_position,
                yticklabels=True)

    # Rest of your formatting code
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize=AXIS_TEXT_SIZE, rotation=70)
    plt.yticks(fontsize=AXIS_TEXT_SIZE, rotation=0)

    # Center xticks by adjusting alignment
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticklabels(),
                       va='bottom',
                       # ha='center',
                       rotation_mode='anchor',
                       position=(0, -0.15)
                       )
    # ax.tick_params(left=False, bottom=False)
    plt.tight_layout()

    # Adjust colorbar label sizes after plotting
    for cbar in ax.figure.axes[-2:]:
        cbar.yaxis.label.set_size(AXIS_TEXT_SIZE-2)
    ax.tick_params(left=False, bottom=False)


def plot_head_pattern_fig_4b(
    model="gpt2",
    experiment="copyVSfact",
    model_folder="gpt2_full",
    domain=None,
    downsampled=False
):
    print("="*100)
    print("Plotting head pattern. Model: ", model, " Experiment: ", experiment, " Model folder: ", model_folder, " Domain: ", domain, " Downsampled: ", downsampled)
    print("="*100)

    # plotting setup
    if experiment == "copyVSfactQnA":
        relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                             "Interrogative", "Relation repeat", "Subject repeat", "Last"]
    else:
        relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                             "Subject repeat", "Relation repeat", "Last"]

    # last token position
    source_position = 13

    # subject and others positions
    if experiment == "copyVSfactQnA":
        source_positions = [1, 4, 5, 6, 7, 8, 9, 13]
        dest_positions = [1, 4, 5, 6, 7, 8, 9, 13]
    else:
        source_positions = [1, 4, 5, 6, 9, 12, 13]
        dest_positions = [1, 4, 5, 6, 9, 12, 13]

    if model == "gpt2":
        if experiment=="copyVSfactDomain":
            layer_pattern = [11, 10, 10, 10, 9, 9, 11, 11]
            head_pattern = [10, 0, 7, 10, 6, 9, 2, 3]
            factual_heads_layer = [11, 10, 11, 11]
            factual_heads_head = [10, 7, 2, 3]
        else:
            layer_pattern = [11, 10, 10, 10, 9, 9]
            head_pattern = [10, 0, 7, 10, 6, 9]
            factual_heads_layer = [11, 10]
            factual_heads_head = [10, 7]
    elif model == "pythia-6.9b":
        layer_pattern = [10, 10, 15, 17, 17, 19, 19, 20, 20, 21, 23]
        head_pattern = [1, 27, 17, 14, 28, 20, 31, 2, 18, 8, 25]
        factual_heads_layer = [21, 20, 17, 10]
        factual_heads_head = [8, 18, 28, 27]
    elif model == "Llama-3.1-8B":
        layer_pattern = [27, 28, 26, 30, 29]
        head_pattern = [20, 15, 13, 12, 10]
        factual_heads_layer = [28, 29]
        factual_heads_head = [15, 10]
    else:
        print("Model not supported!")
        return

    # Load data
    if downsampled:
        data_path = f"../results/{experiment}/head_pattern/{model_folder}_downsampled/head_pattern_data.csv"
    else:
        data_path = f"../results/{experiment}/head_pattern/{model_folder}/head_pattern_data.csv"
    print("Plotting head pattern. Trying to load data from: ", data_path)
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f".csv file now found - {e}")
        return

    directory_path = f"../results/{SAVE_DIR_NAME}/{model}_{experiment}_heads_pattern"
    if domain:
        directory_path = f"../results/{directory_path}/{domain}"
    if downsampled:
        directory_path = f"../results/{directory_path}_downsampled"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Filter and join data
    data_filtered = data[data['source_position'] == source_position]  # last position
    pattern_df = pd.DataFrame({"layer": layer_pattern, "head": head_pattern})
    data_final = pd.merge(data_filtered, pattern_df, on=["layer", "head"])
    # Prepare labels
    data_final["y_label"] = "Layer " + data_final["layer"].astype(str) + " | Head " + data_final["head"].astype(str)

    # Mapping positions
    data_final = data_final[data_final["dest_position"].isin(dest_positions)]
    unique_positions = data_final['dest_position'].unique()
    position_mapping = {position: idx for idx, position in enumerate(unique_positions)}
    data_final["mapped_position"] = data_final['dest_position'].map(position_mapping)

    # Create the heatmap for target and other colors
    data_final["color"] = np.where(data_final['y_label'].isin(
        [f"Layer {layer} | Head {head}" for layer, head in zip(factual_heads_layer, factual_heads_head)]), 'Target',
                                'Other')
    # data_final.sort_values(by=["layer", "head"], ascending=False, axis=0, inplace=True)
    # print(data_final)
    create_heatmap(data_final, relevant_position)
    # Save plot
    plt.savefig(f"{directory_path}/head_pattern_layer.pdf",
                bbox_inches='tight')


    #########################################
    #### Heatmap(s): Layer-Head Combination ####
    #########################################

    # Full position heatmap function
    def create_multiple_heatmaps(data, x, y, fill, title, cmap, ax):
        sns.heatmap(data.pivot_table(index=y, columns=x, values=fill),
                    cmap=cmap, center=0, yticklabels=relevant_position,
                    xticklabels=relevant_position,
                    cbar_kws={'label': 'Attention Score'}, annot=False, fmt='.2f', ax=ax)
        ax.set_title(title, fontsize=AXIS_TEXT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False)
        ax.tick_params('x', rotation=70)
        ax.set_xticklabels(ax.get_xticklabels(),
                        va='bottom',
                        # ha='center',
                        rotation_mode='anchor',
                        position=(0, -0.25)
                        )


    # Full position plot
    if model == "gpt2":
        if experiment == "copyVSfactDomain":
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
        else:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    elif model == "Llama-3.1-8B":
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    plot = None
    for i in range(len(head_pattern)):
        head = head_pattern[i]
        layer = layer_pattern[i]

        # filter positions
        pattern_df = pd.DataFrame({"layer": layer_pattern, "head": head_pattern})
        data_filtered = pd.merge(data, pattern_df, on=["layer", "head"])
        # prepare labels
        data_filtered["y_label"] = "Layer " + data_filtered["layer"].astype(str) + " | Head " + data_filtered["head"].astype(str)

        data_head = data_filtered[(data_filtered["head"] == head) & (data_filtered["layer"] == layer)]
        data_head = data_head[data_head['dest_position'].isin(dest_positions)]
        data_head = data_head[data_head['source_position'].isin(source_positions)]

        # Map positions
        dest_unique_positions = data_head['dest_position'].unique()
        source_unique_positions = data_head['source_position'].unique()

        dest_position_mapping = {position: idx for idx, position in enumerate(dest_unique_positions)}
        source_position_mapping = {position: idx for idx, position in enumerate(source_unique_positions)}

        data_head['dest_mapped'] = data_head['dest_position'].map(dest_position_mapping)
        data_head['source_mapped'] = data_head['source_position'].map(source_position_mapping)

        cmap = FACTUAL_CMAP if (layer in factual_heads_layer and head in factual_heads_head and head!=layer) else COUNTERFACTUAL_CMAP
        create_multiple_heatmaps(data_head, 'dest_mapped', 'source_mapped',
                                    'value', f'Layer {layer} Head {head}', cmap, axes[i])
    # Adjust layout with custom spacing
    if len(layer_pattern) % 2 != 0:
        axes[-1].set_axis_off()
    plt.subplots_adjust(wspace=0.85, hspace=0.95)

    # Saving plot for full position
    plt.savefig(f"{directory_path}/full_pattern.pdf",
                bbox_inches='tight')

    print("Plots saved at: ", directory_path)
    print("=" * 100)
    print("Done plotting head pattern")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and visualize data.')
    parser.add_argument('--model', type=str, nargs='?',
                        help='Name of the model',
                        default="gpt2")
    parser.add_argument('--experiment', type=str, nargs='?',
                        help='Name of the experiment',
                        default="copyVSfact")
    parser.add_argument('--model_folder', type=str, nargs='?',
                        help='Name of the model folder',
                        default="gpt2_full")
    parser.add_argument('--domain', type=str, nargs='?',
                        help='Name of the domain',
                        default=None)
    parser.add_argument('--downsampled', action='store_true',
                        help='Use downsampled dataset',
                        default=False)
    args = parser.parse_args()

    plot_head_pattern_fig_4b(
        model=args.model,
        experiment=args.experiment,
        model_folder=args.model_folder,
        domain=args.domain,
        downsampled=args.downsampled
    )

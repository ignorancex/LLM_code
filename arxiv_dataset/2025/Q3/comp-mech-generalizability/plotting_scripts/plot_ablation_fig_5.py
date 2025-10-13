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


def plot_ablation_figure_5(model="gpt2",
                  experiment="copyVSfact",
                  model_folder="gpt2_full",
                  domain=None,
                  downsampled=False):
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

    #########################################
    ###### Bar Plot: Boosting Heads #########
    #########################################

    # Boosting Heads Bar Plot
    data_long = pd.DataFrame({
        'model': ['GPT2', 'GPT2', 'Pythia-6.9b', 'Pythia-6.9b'],
        'Type': ['Baseline', 'Multiplied Attention\nAltered', 'Baseline', 'Multiplied Attention\nAltered'],
        'Percentage': [4.13, 50.29, 30.32, 49.46]
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="model", y="Percentage", hue="Type",
                data=data_long, palette=[FACTUAL_COLOR, COUNTERFACTUAL_COLOR],
                edgecolor='black', ax=ax)

    # Place legend outside the plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', left=False, bottom=False,
                   labelsize=AXIS_TEXT_SIZE)
    ax.set_xlabel('')
    ax.set_ylabel('% factual answers', fontsize=AXIS_TEXT_SIZE)
    plt.tight_layout()

    # Save bar plot
    plt.savefig(f"{directory_path}/multiplied_pattern.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("Plots saved at: ", directory_path)
    print("=" * 100)
    print("Done plotting ablation results")
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

    plot_ablation_figure_5(
        model=args.model,
        experiment=args.experiment,
        model_folder=args.model_folder,
        domain=args.domain,
        downsampled=args.downsampled
    )
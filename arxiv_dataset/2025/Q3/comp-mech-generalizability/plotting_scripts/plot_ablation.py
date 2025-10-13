import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
def create_heatmap(data, x, y, fill, title, save_path=None):
    """
    Create a heatmap using seaborn.
    """
    plt.figure(figsize=(10, 8))
    # Aggregate to handle duplicate combinations of x and y
    # data = data.groupby([x, y], as_index=False)[fill].mean()
    data = data.groupby([x, y], as_index=False, observed=False)[fill].mean()
    pivot_table = data.pivot(index=y, columns=x, values=fill)

    ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    ax.set_title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def process_component(data, component_name, x, y, std_dev, folder_name):
    """
    Process a specific component type and generate heatmaps.
    """
    # Filter rows for the specified component
    component_data = data[data['component'].str.contains(component_name, na=False)].copy()

    # Handle missing or non-numeric values in 'layer'
    component_data['layer'] = pd.to_numeric(component_data['layer'], errors='coerce')
    component_data.dropna(subset=['layer'], inplace=True)  # Drop rows with NaN in 'layer'
    if component_data.empty:
        print(f"No valid data for component '{component_name}'. Skipping...")
        return

    # Compute max_layer
    max_layer = component_data['layer'].max()

    # Handle missing or non-numeric values in 'x' (e.g., 'position' or 'head')
    component_data[x] = pd.to_numeric(component_data[x], errors='coerce')
    component_data.dropna(subset=[x], inplace=True)
    if component_data.empty:
        print(f"No valid data for component '{component_name}' after handling '{x}'. Skipping...")
        return

    # Compute max_position
    max_position = component_data[x].max()

    # Convert 'layer' and 'x' to categorical
    component_data['layer'] = pd.Categorical(component_data['layer'], categories=range(int(max_layer) + 1))
    component_data[x] = pd.Categorical(component_data[x], categories=range(int(max_position) + 1))

    # Generate heatmaps
    for metric in ['mem', 'cp', 'diff']:
        create_heatmap(
            component_data,
            x,
            'layer',
            metric,
            f"{component_name.capitalize()} ablation - {metric}",
            save_path=os.path.join(folder_name, f"{component_name}_{metric}.pdf")
        )

        if std_dev:
            create_heatmap(
                component_data,
                x,
                'layer',
                f"{metric}_std",
                f"{component_name.capitalize()} ablation - {metric} std",
                save_path=os.path.join(folder_name, f"{component_name}_{metric}_std.pdf")
            )

def plot_ablation(
    model="gpt2",
    experiment="copyVSfact",
    model_folder="gpt2_full",
    std_dev=1,
    folder_name="python_paper_plots"
):
    # List files in the folder
    files = os.listdir(folder_name)

    # Process the CSV files based on their presence
    if "ablation_data.csv" in files:
        data = pd.read_csv(os.path.join(folder_name, "ablation_data.csv"))
        for component in ['mlp_out', 'attn_out', 'resid_pre', 'head']:
            process_component(data, component, 'position' if component != 'head' else 'head', 'layer', std_dev, folder_name)
    else:
        component_files = {
            "ablation_data_mlp_out.csv": 'mlp_out',
            "ablation_data_attn_out.csv": 'attn_out',
            "ablation_data_resid_pre.csv": 'resid_pre',
            "ablation_data_head.csv": 'head'
        }
        for file, component in component_files.items():
            if file in files:
                data = pd.read_csv(os.path.join(folder_name, file))
                process_component(data, component, 'position' if component != 'head' else 'head', 'layer', std_dev, folder_name)
    pass

def main(folder_name, std_dev):
    plot_ablation(folder_name=folder_name, std_dev=std_dev)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script_name.py <folder_name> <std_dev>")
    #     sys.exit(1)
    #
    # folder_name = sys.argv[1]
    # std_dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    folder_name = "../results/copyVSfact/ablation/gpt2_full/"
    std_dev = 1
    parser = argparse.ArgumentParser(description='Process and visualize data.')
    parser.add_argument('folder_name', type=str, nargs='?',
                        help='Path to the folder containing the data',
                        default=folder_name)
    parser.add_argument('std_dev', type=int, nargs='?',
                        help='Standard deviation',
                        default=std_dev)
    args = parser.parse_args()
    plot_ablation(
        folder_name=args.folder_name,
        std_dev=args.std_dev
    )
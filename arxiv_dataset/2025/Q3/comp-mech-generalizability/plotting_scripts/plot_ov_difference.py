import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def read_and_prepare_data(folder_name):
    """
    Method to read data and prepare the data
    """
    data = pd.read_csv(f"{folder_name}/ov_difference_data.csv")

    selected_combinations = pd.DataFrame({
        'layer': [11, 10, 10, 10, 9, 9],
        'head': [10, 0, 7, 10, 6, 9]
    })
    data['combination'] = data.apply(lambda row: f"Layer {int(row['layer'])} Head {int(row['head'])}", axis=1)
    selected_combinations_str = [f"Layer {l} Head {h}" for l, h in
                                 zip(selected_combinations['layer'], selected_combinations['head'])]

    filtered_data = data[data['combination'].isin(selected_combinations_str)]
    print(filtered_data)

    return filtered_data


# Function to generate scatter plot
def create_scatter_plot(data_sampled, folder_name):
    if data_sampled.empty:
        print("No data to plot after sampling.")
        return  # Exit the function if no data is available

    data_sampled['subtitle'] = data_sampled.apply(lambda row: f"Layer {row['layer']} Head {row['head']}", axis=1)

    # If there are no valid 'subtitle' values, avoid plotting
    if data_sampled['subtitle'].nunique() == 0:
        print("No valid 'subtitle' values found for plotting.")
        return

    plt.figure(figsize=(10, 14))
    g = sns.FacetGrid(data_sampled, col='subtitle', col_wrap=2, height=5)
    g.map(sns.scatterplot, 'mem_input', 'cp_input', color='#357EDD', alpha=0.8)

    g.set_axis_labels('Source token = Subject', 'Source token = Altered')
    g.set_titles(col_template='{col_name}')
    g.set(xticks=range(0, 21, 5), yticks=range(-20, 21, 5))

    plt.tight_layout()
    plt.savefig(f"{folder_name}/copyVSfact_ov_scatterplot.pdf")


def create_histogram(data, folder_name):
    """
    Function to generate histogram
    """
    filtered_data = data[(data['layer'] == 11) & (data['head'] == 10)]
    filtered_data['difference'] = filtered_data['mem_input'] - filtered_data['cp_input']

    plt.figure(figsize=(8, 6))
    sns.histplot(filtered_data['difference'], bins=30, kde=False, color='blue')
    plt.title('Histogram of Differences between mem_input and cp_input')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{folder_name}/difference_histogram.pdf")


def plot_ov_difference(
    model="gpt2",
    experiment="copyVSfact",
    model_folder="gpt2_full"
):
    folder_name = f"{experiment}/ov_difference/{model_folder}"

    # Read and prepare data
    data = read_and_prepare_data(folder_name)

    # Sample the data and create the scatter plot
    data_sampled = data.groupby('combination').sample(frac=0.3, random_state=42).reset_index(drop=True)
    create_scatter_plot(data_sampled, folder_name)

    # Create the histogram plot
    create_histogram(data, folder_name)
        
# Main function
def main():
    folder_name = '../results/copyVSfact/ov_difference/gpt2_full'
    parser = argparse.ArgumentParser(description='Process and visualize data.')
    parser.add_argument('folder_name', type=str, nargs='?',
                        help='Path to the folder containing the data',
                        default=folder_name)
    args = parser.parse_args()

    folder_name = args.folder_name

    # Read and prepare data
    data = read_and_prepare_data(folder_name)

    # Sample the data and create the scatter plot
    data_sampled = data.groupby('combination').sample(frac=0.3, random_state=42).reset_index(drop=True)
    create_scatter_plot(data_sampled, folder_name)

    # Create the histogram plot
    create_histogram(data, folder_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and visualize data.')
    parser.add_argument('model', type=str, nargs='?',
                        help='Name of the model',
                        default="gpt2")
    parser.add_argument('experiment', type=str, nargs='?',
                        help='Name of the experiment',
                        default="copyVSfact")
    parser.add_argument('model_folder', type=str, nargs='?',
                        help='Name of the model folder',
                        default="gpt2_full")
    args = parser.parse_args()
    plot_ov_difference(
        model=args.model,
        experiment=args.experiment,
        model_folder=args.model_folder
    )

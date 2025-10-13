import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ['MPRAGE', 'BLOCH', 'BLOCH-PAIRED']
MODEL_NAMES = {
    'MPRAGE': 'Baseline',
    'BLOCH': 'Sequence-augmented',
    'BLOCH-PAIRED': 'Sequence-invariant'
}

SITE_NAMES = {
    'guys': 'GST',
    'hh': 'HH',
    'iop': 'IOP'
}

def get_results_df(results_dir):
    results_file = os.path.join(results_dir, "test_results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        print(f"results_dir: {results_dir}")
        run_name = results_dir.split("/")[0]
        print(f"run_name: {run_name}")
        data_pc = int(run_name.split("pc")[1])
        df["% Training Data"] = data_pc
        method = run_name.split("simclr-")[1].split("-pc")[0]
        df["Method"] = MODEL_NAMES.get(method.upper(), method.upper())
        return df
    else:
        print(f"Results file not found: {results_file}")
        return None
    
def spider_plot(results_df, metric="DSC"):
    # Calculate mean metric for each modality dataset and method
    spider_data = results_df.groupby(['Modality Dataset', 'Method'])[metric].mean().unstack()
    
    # Set up the angles for the spider plot
    angles = np.linspace(0, 2*np.pi, len(spider_data.index), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the plot
    
    # Calculate min and max values for smart limits, filtering out nans
    min_val = spider_data.values[~np.isnan(spider_data.values)].min()
    max_val = spider_data.values[~np.isnan(spider_data.values)].max()
    
    # Set limits to 95% of min and 105% of max
    ylim_min = min_val * 0.95
    ylim_max = max_val * 1.05
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    for method in spider_data.columns:
        values = spider_data[method].values
        values = np.concatenate((values, [values[0]]))  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set the ylim
    ax.set_ylim(ylim_min, ylim_max)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spider_data.index, rotation=45)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.title(f"Mean {metric} by Modality Dataset and Method")
    plt.tight_layout()
    return fig

# Set style for fancy plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

# List of model configurations to check
results_dirs = glob.glob("spm-*-cnn-simclr-*/")

# Create results directory
plot_dir = "plots/healthy_segmentation"
os.makedirs(plot_dir, exist_ok=True)

# Collect all results into single dataframe
all_results = []
for results_dir in results_dirs:
    df = get_results_df(results_dir)
    if df is not None:
        all_results.append(df)

if not all_results:
    print("No results found!")
    exit()

results_df = pd.concat(all_results, ignore_index=True)

# Tidy up names of columns
results_df.rename(columns={"dice": "DSC", "hd95": "HD95", "class": "Class", "modality": "Modality", "dataset": "Dataset", "site": "Site"}, inplace=True)

# Multiply DSC by 100 to get percentage
results_df["DSC"] = results_df["DSC"] * 100

# Map site names
results_df["Site"] = results_df["Site"].map(SITE_NAMES)

# Create a merged column for modality and dataset
results_df["Modality Dataset"] = results_df["Site"] + " [" + results_df["Modality"] + "]"

# Set categorical order for 'Modality' and 'Site'
modality_order = ['T1w', 'T2w', 'PDw']
site_order = ['GST', 'HH', 'IOP']

# Ensure 'Modality' and 'Site' columns are categorical with the specified order
results_df['Modality'] = pd.Categorical(results_df['Modality'], categories=modality_order, ordered=True)
results_df['Site'] = pd.Categorical(results_df['Site'], categories=site_order, ordered=True)

# Recreate 'Modality Dataset' column to reflect the new order
results_df["Modality Dataset"] = results_df["Site"].astype(str) + " [" + results_df["Modality"].astype(str) + "]"

# Sort the DataFrame based on the new categorical order
results_df.sort_values(by=['Site', 'Modality'], inplace=True)

# Get unique percentages
training_percentages = sorted(results_df["% Training Data"].unique(), reverse=True)

print(results_df.head())

# Loop over percentages
for percentage in training_percentages:
    # Create subdirectory for this percentage
    percentage_dir = os.path.join(plot_dir, f"{percentage}pc")
    os.makedirs(percentage_dir, exist_ok=True)
    
    # Filter data for this percentage
    percentage_df = results_df[results_df["% Training Data"] == percentage]
    
    # Get unique classes for this percentage
    classes = percentage_df['Class'].unique()
    
    # Generate plots for all data and per-class
    for class_name in [None] + list(classes):
        class_suffix = f"_{class_name.lower()}" if class_name else ""
        class_data = percentage_df[percentage_df['Class'] == class_name] if class_name else percentage_df
        
        # 1. Boxplot of Dice scores by model and dataset
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=class_data, x="Modality Dataset", y="DSC", hue="Method",
                         hue_order=[MODEL_NAMES[m] for m in MODEL_ORDER],
                         boxprops={'alpha': 0.8, 'linewidth': 2},
                         showfliers=False,
                         width=0.8)
        plt.title(f"Dice Similarity Coefficient by Modality and Dataset ({percentage}% Training Data)", 
                 pad=20, fontsize=16, fontweight='bold')
        plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
        plt.ylabel("DSC (%)", fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(percentage_dir, f"dice_boxplot_by_modality_dataset{class_suffix}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Violin plot of Dice scores by model and dataset
        plt.figure(figsize=(15, 8))
        ax = sns.violinplot(data=class_data, x="Modality Dataset", y="DSC", hue="Method",
                            hue_order=[MODEL_NAMES[m] for m in MODEL_ORDER],
                            density_norm='width', inner='quartile', alpha=0.5,
                            cut=0,
                            )
        plt.title(f"Dice Similarity Coefficient by Modality and Dataset ({percentage}% Training Data)", 
                 pad=20, fontsize=16, fontweight='bold')
        plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
        plt.ylabel("DSC (%)", fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(percentage_dir, f"dice_violinplot_by_modality_dataset{class_suffix}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Boxplot of HD95 scores by model and dataset
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=class_data, x="Modality Dataset", y="HD95", hue="Method",
                         hue_order=[MODEL_NAMES[m] for m in MODEL_ORDER],
                         boxprops={'alpha': 0.8, 'linewidth': 2},
                         showfliers=False,
                         width=0.8)
        plt.title(f"95% Hausdorff Distance by Modality and Dataset ({percentage}% Training Data)", 
                 pad=20, fontsize=16, fontweight='bold')
        plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
        plt.ylabel("HD95 (mm)", fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(percentage_dir, f"hd95_boxplot_by_modality_dataset{class_suffix}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Violin plot of HD95 scores by model and dataset
        plt.figure(figsize=(15, 8))
        ax = sns.violinplot(data=class_data, x="Modality Dataset", y="HD95", hue="Method",
                            hue_order=[MODEL_NAMES[m] for m in MODEL_ORDER],
                            density_norm='width', inner='quartile', alpha=0.5,
                            cut=0,
                            )
        plt.title(f"95% Hausdorff Distance by Modality and Dataset ({percentage}% Training Data)", 
                 pad=20, fontsize=16, fontweight='bold')
        plt.xlabel("Modality Dataset", fontsize=14, labelpad=15)
        plt.ylabel("HD95 (mm)", fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Method", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(percentage_dir, f"hd95_violinplot_by_modality_dataset{class_suffix}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Spider plot of Dice scores by modality and dataset
        plt.figure(figsize=(10, 10))
        spider_plot(class_data, metric="DSC")
        plt.savefig(os.path.join(percentage_dir, f"dice_spider_plot{class_suffix}.png"))
        plt.close()

        # 6. Spider plot of HD95 scores by modality and dataset
        plt.figure(figsize=(10, 10))
        spider_plot(class_data, metric="HD95")
        plt.savefig(os.path.join(percentage_dir, f"hd95_spider_plot{class_suffix}.png"))
        plt.close()

        # 7. Summary statistics table
        summary_stats = class_data.groupby(['Modality Dataset', 'Method'])[['DSC', 'HD95']].agg(['mean', 'std', 'median', 'min', 'max', 'sem']).round(1)
        if class_name:
            summary_stats.to_csv(os.path.join(percentage_dir, f"summary_statistics_{class_name.lower()}.csv"))
        else:
            summary_stats.to_csv(os.path.join(percentage_dir, "summary_statistics.csv"))

        # Print summary for this percentage and class
        print(f"\nResults Summary for {percentage}% Training Data{' and ' + class_name if class_name else ''}:")
        print(summary_stats)

print(f"\nPlots saved in: {plot_dir}")

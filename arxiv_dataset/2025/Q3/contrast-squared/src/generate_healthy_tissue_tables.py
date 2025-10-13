import os
import pandas as pd
import numpy as np
import glob

MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']
SITES = ['GST', 'HH', 'IOP']
MODALITIES = ['T1w', 'T2w', 'PDw']
PERCENTAGES = ['1pc', '10pc', '100pc']
# These are the tissue types we'll look for in the files
TISSUES = ['white_matter', 'gray_matter', 'csf']

# Define which metrics should be minimized vs maximized
METRIC_ORDERING = {
    'DSC': 'maximize',
    'HD95': 'minimize'
}

def format_mean_std(mean, std, is_best=False, is_second=False):
    """Format mean ± std with appropriate precision and highlighting"""
    if np.isnan(mean) or np.isnan(std):
        return "---"
    
    if mean < 0.01:
        formatted = f"{mean:.2e} ± {std:.2e}"
    elif mean < 1:
        formatted = f"{mean:.3f} ± {std:.3f}"
    else:
        formatted = f"{mean:.2f} ± {std:.2f}"
    
    if is_best:
        formatted = "\\textbf{" + formatted + "}"
    elif is_second:
        formatted = "\\underline{" + formatted + "}"
    
    return formatted

def get_ranking_indices(values, metric):
    """Get indices of best and second best values, handling NaN"""
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    if not valid_indices:
        return None, None
    
    # Determine sort order based on metric
    reverse = METRIC_ORDERING.get(metric, 'maximize') == 'maximize'
    
    sorted_indices = sorted(valid_indices, key=lambda i: values[i], reverse=reverse)
    best_idx = sorted_indices[0] if len(sorted_indices) > 0 else None
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    
    return best_idx, second_idx

def create_combined_tissue_table(all_tissue_data, tissue, metric):
    """Create a LaTeX table for a specific tissue type combining all percentages"""
    print(f"\nCreating combined table for {tissue} tissue - {metric}")
    
    # Check if we have data for this tissue
    if tissue not in all_tissue_data:
        print(f"No data found for {tissue}")
        return None
    
    tissue_data = all_tissue_data[tissue]
    
    # Define shortened model names
    MODEL_NAMES = {
        'Baseline': 'Base',
        'Sequence-augmented': 'Seq-aug',
        'Sequence-invariant': 'Seq-inv'
    }
    
    # Create descriptive caption
    metric_descriptions = {
        'DSC': 'Dice Similarity Coefficient (higher is better)',
        'HD95': '95th percentile Hausdorff Distance in mm (lower is better)'
    }
    
    # Convert underscore to space for display
    display_tissue = tissue.replace('_', ' ')
    
    caption = f"{display_tissue.title()} segmentation performance using {metric_descriptions.get(metric, metric)} across different training data percentages. "
    caption += "Values show mean ± standard error, with \\textbf{bold} and \\underline{underlined} indicating best and second-best results for each dataset and percentage. "
    caption += "GST represents the training domain."
    
    # Calculate total columns (3 models per percentage + 1 for dataset)
    total_cols = len(PERCENTAGES) * len(MODEL_ORDER) + 1
    
    # Create tabular format with vertical lines between percentage groups
    tabular_format = "l" + ("c" * len(MODEL_ORDER) + "|") * (len(PERCENTAGES) - 1) + "c" * len(MODEL_ORDER)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:healthy_{tissue}_{metric.lower()}_combined_results}}",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{tabular_format}}}",
        "\\toprule"
    ]

    # Create more descriptive percentage headers with bold
    header = ["& \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{\\textbf{" + str(int(pc.replace('pc',''))) + "\\% Training Data}}" 
             for pc in PERCENTAGES]
    latex_lines.append(" " + " ".join(header) + " \\\\")
    
    # Add bold to shortened model names
    bold_models = [f"\\textbf{{{MODEL_NAMES[model]}}}" for model in MODEL_ORDER]
    model_line = "& " + " & ".join(bold_models * len(PERCENTAGES)) + " \\\\"
    latex_lines.extend([
        f"\\cmidrule(lr){{2-{total_cols}}}",
        model_line,
        "\\midrule"
    ])

    # Process datasets
    current_domain = None
    ordered_datasets = []
    for site in SITES:
        for modality in MODALITIES:
            ordered_datasets.append(f"{site} [{modality}]")
    
    for dataset in ordered_datasets:
        # Check if this dataset exists in any of the percentage data
        dataset_exists = False
        for pc in PERCENTAGES:
            if pc in tissue_data and dataset in tissue_data[pc].index.get_level_values(0).unique():
                dataset_exists = True
                break
                
        if not dataset_exists:
            continue
            
        print(f"\nDataset: {dataset}")
        
        # Add domain header if domain changes
        domain = "In Domain" if "GST" in dataset else "Out of Domain"
        if domain != current_domain:
            current_domain = domain
            if domain == "Out of Domain":
                latex_lines.append("\\midrule")
            latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{domain}}}}} \\\\")
            latex_lines.append("\\midrule")
        
        # Make dataset names bold
        bold_dataset = f"\\textbf{{{dataset}}}"
        
        row_values = []
        
        # Process each percentage
        for pc in PERCENTAGES:
            if pc not in tissue_data or dataset not in tissue_data[pc].index.get_level_values(0).unique():
                # If this dataset doesn't exist for this percentage, add empty cells
                row_values.extend(["---"] * len(MODEL_ORDER))
                continue
                
            pc_data = tissue_data[pc]
            
            # Get values for ranking
            values = []
            for model in MODEL_ORDER:
                try:
                    mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                    values.append(mean)
                except:
                    values.append(np.nan)
            
            best_idx, second_idx = get_ranking_indices(values, metric)
            
            # Format values for this percentage
            for i, model in enumerate(MODEL_ORDER):
                try:
                    mean = pc_data.loc[dataset].loc[model][(metric, 'mean')]
                    std = pc_data.loc[dataset].loc[model][(metric, 'sem')]
                    is_best = (i == best_idx)
                    is_second = (i == second_idx)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    row_values.append(formatted)
                except Exception as e:
                    print(f"    Error for {model} in {pc}: {str(e)}")
                    row_values.append("---")
                
        latex_lines.append(f"{bold_dataset} & " + " & ".join(row_values) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    # Healthy segmentation directory
    task_dir = os.path.join('plots', 'healthy_segmentation')
    if not os.path.exists(task_dir):
        print(f"No results directory found for healthy segmentation")
        return

    # Create a nested structure to store all tissue data by tissue type and percentage
    all_tissue_data = {}
    
    # First, find all tissue-specific summary files
    print("\nLooking for tissue-specific summary files:")
    
    # Find all tissue-specific files
    found_tissues = set()
    for pc in PERCENTAGES:
        pc_dir = os.path.join(task_dir, pc)
        if not os.path.exists(pc_dir):
            print(f"  Directory not found: {pc_dir}")
            continue
            
        # Use glob to find all tissue-specific summary files
        tissue_files = glob.glob(os.path.join(pc_dir, "summary_statistics_*.csv"))
        
        for file_path in tissue_files:
            file_name = os.path.basename(file_path)
            # Extract the tissue name from the file name
            if file_name.startswith("summary_statistics_") and file_name.endswith(".csv"):
                tissue_name = file_name[len("summary_statistics_"):-4]  # Remove prefix and suffix
                found_tissues.add(tissue_name)
                print(f"  Found file for tissue: {tissue_name} in {pc}")
    
    # If no tissue files found, check if the specified tissue types exist
    if not found_tissues:
        for tissue in TISSUES:
            for pc in PERCENTAGES:
                tissue_file = os.path.join(task_dir, pc, f"summary_statistics_{tissue}.csv")
                if os.path.exists(tissue_file):
                    found_tissues.add(tissue)
                    print(f"  Found file for specified tissue: {tissue} in {pc}")
    
    if not found_tissues:
        print("No tissue-specific files found!")
        return
    
    print(f"\nFound files for these tissues: {found_tissues}")
    
    # Now load the data for each tissue
    for tissue in found_tissues:
        all_tissue_data[tissue] = {}
        
        for pc in PERCENTAGES:
            tissue_file = os.path.join(task_dir, pc, f"summary_statistics_{tissue}.csv")
            if os.path.exists(tissue_file):
                print(f"\nLoading file: {tissue_file}")
                try:
                    df = pd.read_csv(tissue_file, header=[0,1], index_col=[0,1])
                    print(f"DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {df.columns}")
                    all_tissue_data[tissue][pc] = df
                    print(f"  Successfully loaded data for {tissue} from {pc}")
                except Exception as e:
                    print(f"  Error loading {tissue_file}: {e}")
    
    # Print what we found
    print("\nSummary of tissue data loaded:")
    for tissue in all_tissue_data:
        percentages_found = list(all_tissue_data[tissue].keys())
        if percentages_found:
            print(f"  {tissue}: Found data for percentages: {percentages_found}")
        else:
            print(f"  {tissue}: No data found for any percentage")
    
    # Get metrics from any available dataframe
    metrics = []
    for tissue in all_tissue_data:
        for pc in all_tissue_data[tissue]:
            sample_df = all_tissue_data[tissue][pc]
            metrics = sample_df.columns.get_level_values(0).unique()
            print(f"\nMetrics found for {tissue}: {metrics}")
    
    # Generate combined tables for each tissue and metric
    for tissue in all_tissue_data:        
        # Create a subfolder for each tissue type
        tissue_dir = os.path.join(task_dir, tissue)
        os.makedirs(tissue_dir, exist_ok=True)
        
        for metric in metrics:
            latex_table = create_combined_tissue_table(all_tissue_data, tissue, metric)
            if latex_table:
                output_file = os.path.join(tissue_dir, f'{metric.lower()}_combined_table.txt')
                with open(output_file, 'w') as f:
                    f.write(latex_table)
                print(f"Generated combined table for {tissue} - {metric}")

if __name__ == "__main__":
    main() 
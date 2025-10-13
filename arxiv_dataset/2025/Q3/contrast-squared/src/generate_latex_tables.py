import os
import pandas as pd
import numpy as np

TASKS = ['healthy_segmentation', 'denoise', 'stroke_segmentation']
PERCENTAGES = ['1pc', '10pc', '100pc']
MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']
SITES = ['GST', 'HH', 'IOP']
MODALITIES = ['T1w', 'T2w', 'PDw']

# Define which metrics should be minimized vs maximized
METRIC_ORDERING = {
    'DSC': 'maximize',
    'PSNR': 'maximize',
    'MSE': 'minimize',
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
        formatted = f"{mean:.1f} ± {std:.1f}"
    
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

def create_latex_table(all_data, metric, task):
    """Create a LaTeX table for a given metric"""
    print(f"\nCreating table for {task} - {metric}")
    
    total_cols = len(PERCENTAGES) * len(MODEL_ORDER) + 1  # +1 for dataset column
    
    # Define shortened model names
    MODEL_NAMES = {
        'Baseline': 'Base',
        'Sequence-augmented': 'Seq-aug',
        'Sequence-invariant': 'Seq-inv'
    }
    
    # Define sites and modalities based on task
    if task == 'stroke_segmentation':
        SITES = ['ARC']
        MODALITIES = ['T1w', 'T2w', 'FLAIR']
    else:
        SITES = ['GST', 'HH', 'IOP']
        MODALITIES = ['T1w', 'T2w', 'PDw']
    
    # Create descriptive caption based on metric
    metric_descriptions = {
        'DSC': 'Dice Similarity Coefficient (higher is better)',
        'HD95': '95th percentile Hausdorff Distance in mm (lower is better)',
        'PSNR': 'Peak Signal-to-Noise Ratio in dB (higher is better)',
        'MSE': 'Mean Squared Error (lower is better)'
    }
    
    task_descriptions = {
        'healthy_segmentation': 'Healthy brain tissue segmentation',
        'stroke_segmentation': 'Stroke lesion segmentation',
        'denoise': 'Image denoising'
    }
    
    # Create caption text
    caption = f"{task_descriptions.get(task, task)} performance using {metric_descriptions.get(metric, metric)}. "
    caption += "Values show mean ± standard error, with \\textbf{bold} and \\underline{underlined} indicating best and second-best results. "
    caption += "GST represents the training domain."
    
    # Create tabular format with vertical lines between percentage groups
    tabular_format = "l" + ("c" * len(MODEL_ORDER) + "|") * (len(PERCENTAGES) - 1) + "c" * len(MODEL_ORDER)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{task}_{metric.lower()}_results}}",
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
        f"\\cmidrule(lr){{2-{len(PERCENTAGES) * len(MODEL_ORDER) + 1}}}",
        model_line,
        "\\midrule"
    ])

    print("\nProcessing datasets:")
    ordered_datasets = []
    for site in SITES:
        for modality in MODALITIES:
            ordered_datasets.append(f"{site} [{modality}]")
    
    current_domain = None
    for dataset in ordered_datasets:
        if dataset not in all_data[PERCENTAGES[-1]].index.get_level_values(0).unique():
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
        for pc in PERCENTAGES:
            print(f"  Processing {pc}")
            df = all_data[pc]
            
            values = []
            for model in MODEL_ORDER:
                try:
                    mean = df.loc[dataset].loc[model][(metric, 'mean')]
                    values.append(mean)
                except:
                    values.append(np.nan)
            
            best_idx, second_idx = get_ranking_indices(values, metric)
            
            for i, model in enumerate(MODEL_ORDER):
                try:
                    mean = df.loc[dataset].loc[model][(metric, 'mean')]
                    std = df.loc[dataset].loc[model][(metric, 'sem')]
                    is_best = (i == best_idx)
                    is_second = (i == second_idx)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    print(f"    {model}: {formatted}")
                    row_values.append(formatted)
                except Exception as e:
                    print(f"    Error for {model}: {str(e)}")
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
    for task in TASKS:
        print(f"\nProcessing task: {task}")
        task_dir = os.path.join('plots', task)
        if not os.path.exists(task_dir):
            print(f"No results directory found for {task}")
            continue

        all_data = {}
        
        for pc in PERCENTAGES:
            stats_file = os.path.join(task_dir, pc, 'summary_statistics.csv')
            if os.path.exists(stats_file):
                print(f"\nLoading {stats_file}")
                df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
                print("DataFrame shape:", df.shape)
                print("DataFrame columns:", df.columns)
                all_data[pc] = df
            else:
                print(f"No summary statistics found for {task} {pc}")
                continue

        if not all_data:
            continue

        metrics = all_data[PERCENTAGES[0]].columns.get_level_values(0).unique()
        print(f"\nMetrics found: {metrics}")
        
        for metric in metrics:
            latex_table = create_latex_table(all_data, metric, task)
            
            output_file = os.path.join(task_dir, f'{metric.lower()}_table.txt')
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"Generated table for {task} {metric}")

if __name__ == "__main__":
    main() 
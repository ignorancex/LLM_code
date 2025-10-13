import os
import pandas as pd
import numpy as np

MODEL_ORDER = ['Baseline', 'Sequence-augmented', 'Sequence-invariant']
SITES = ['ARC']
MODALITIES = ['T1w', 'T2w', 'FLAIR']

# Define shortened model names
MODEL_NAMES = {
    'Baseline': 'Base',
    'Sequence-augmented': 'Seq-aug',
    'Sequence-invariant': 'Seq-inv'
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
    reverse = metric == 'DSC'  # DSC higher is better, HD95 lower is better
    
    sorted_indices = sorted(valid_indices, key=lambda i: values[i], reverse=reverse)
    best_idx = sorted_indices[0] if len(sorted_indices) > 0 else None
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    
    return best_idx, second_idx

def create_stroke_table(dsc_data, hd_data):
    """Create a LaTeX table combining DSC and HD95 results"""
    print("\nCreating stroke table...")
    print(f"Data shape: {dsc_data.shape}")
    print("\nData index levels:")
    print(dsc_data.index.levels)
    print("\nData columns:")
    print(dsc_data.columns)
    
    total_cols = len(MODEL_ORDER) * 2 + 1
    
    caption = "Stroke lesion segmentation performance using 100\\% training data. "
    caption += "Values show mean ± standard error, with \\textbf{bold} and \\underline{underlined} indicating best and second-best results for each metric. "
    caption += "DSC (higher is better) and HD95 in mm (lower is better) are shown for each model."
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:stroke_results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l" + "c" * (total_cols-1) + "}",
        "\\toprule"
    ]

    # Create headers
    metric_header = "& \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{\\textbf{DSC}} & \\multicolumn{" + str(len(MODEL_ORDER)) + "}{c}{\\textbf{HD95 (mm)}} \\\\"
    latex_lines.append(metric_header)
    
    # Add bold model names
    bold_models = [f"\\textbf{{{MODEL_NAMES[model]}}}" for model in MODEL_ORDER]
    model_line = "& " + " & ".join(bold_models) + " & " + " & ".join(bold_models) + " \\\\"
    latex_lines.extend([
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        model_line,
        "\\midrule"
    ])

    # Process each dataset
    for site in SITES:
        for modality in MODALITIES:
            dataset = f"{site} [{modality}]"
            print(f"\nProcessing dataset: {dataset}")
            bold_dataset = f"\\textbf{{{dataset}}}"
            
            # Process DSC values
            dsc_values = []
            for model in MODEL_ORDER:
                try:
                    mean = dsc_data.loc[dataset].loc[model][('DSC', 'mean')]
                    dsc_values.append(mean)
                    print(f"DSC - {model}: {mean}")
                except Exception as e:
                    print(f"Error getting DSC for {model}: {str(e)}")
                    dsc_values.append(np.nan)
            
            best_dsc, second_dsc = get_ranking_indices(dsc_values, 'DSC')
            print(f"DSC rankings - Best: {best_dsc}, Second: {second_dsc}")
            
            # Process HD95 values
            hd_values = []
            for model in MODEL_ORDER:
                try:
                    mean = hd_data.loc[dataset].loc[model][('HD95', 'mean')]
                    hd_values.append(mean)
                    print(f"HD95 - {model}: {mean}")
                except Exception as e:
                    print(f"Error getting HD95 for {model}: {str(e)}")
                    hd_values.append(np.nan)
            
            best_hd, second_hd = get_ranking_indices(hd_values, 'HD95')
            print(f"HD95 rankings - Best: {best_hd}, Second: {second_hd}")
            
            # Format all values
            row_values = []
            for i, model in enumerate(MODEL_ORDER):
                # Add DSC value
                try:
                    mean = dsc_data.loc[dataset].loc[model][('DSC', 'mean')]
                    std = dsc_data.loc[dataset].loc[model][('DSC', 'sem')]
                    is_best = (i == best_dsc)
                    is_second = (i == second_dsc)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    print(f"Formatted DSC - {model}: {formatted}")
                    row_values.append(formatted)
                except Exception as e:
                    print(f"Error formatting DSC for {model}: {str(e)}")
                    row_values.append("---")
                
            for i, model in enumerate(MODEL_ORDER):
                # Add HD95 value
                try:
                    mean = hd_data.loc[dataset].loc[model][('HD95', 'mean')]
                    std = hd_data.loc[dataset].loc[model][('HD95', 'sem')]
                    is_best = (i == best_hd)
                    is_second = (i == second_hd)
                    formatted = format_mean_std(mean, std, is_best, is_second)
                    print(f"Formatted HD95 - {model}: {formatted}")
                    row_values.append(formatted)
                except Exception as e:
                    print(f"Error formatting HD95 for {model}: {str(e)}")
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
    task_dir = os.path.join('plots', 'stroke_segmentation', '100pc')
    print(f"\nLooking for data in: {task_dir}")
    
    # Load DSC and HD95 data
    stats_file = os.path.join(task_dir, 'summary_statistics.csv')
    print(f"Stats file path: {stats_file}")
    
    if os.path.exists(stats_file):
        print("Found statistics file")
        df = pd.read_csv(stats_file, header=[0,1], index_col=[0,1])
        print("\nLoaded DataFrame:")
        print(df.head())
        
        # Generate combined table
        latex_table = create_stroke_table(df, df)
        
        # Save the table
        output_file = os.path.join('plots', 'stroke_segmentation', 'combined_table.txt')
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nGenerated combined stroke results table at: {output_file}")
    else:
        print(f"No summary statistics found at {stats_file}")

if __name__ == "__main__":
    main() 
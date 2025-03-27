import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product  # For combinations of bits and model_types


def main():
    # Set the experiment number
    EXP = 1  # Or 1, 2, 3, depending on your needs

    # Get experiment settings
    targets, plot_directory, experiments, ks, model_types, bits_list_per_target = get_experiment_settings(EXP)

    # Base directory for the log files
    base_dir = '/scratch/qilong3/transferattack/results'

    # Read and parse logs
    df = read_and_parse_logs(base_dir, targets, experiments, bits_list_per_target, model_types, ks, EXP)

    # Process dataframe
    df = process_dataframe(df)

    # Create plots
    create_plots(df, EXP, plot_directory, bits_list_per_target, experiments)

    print(f"Plots have been saved to the {plot_directory} directory.")



def create_plots(df, EXP, plot_directory, bits_list_per_target, experiments):
    # Create 'plots' directory if it doesn't exist
    os.makedirs(plot_directory, exist_ok=True)

    # List of metric columns to plot
    metric_columns = df.columns.difference(['target', 'k', 'experiment', 'bits', 'model_type', 'source'])

    # Update metric_columns to include 'Evasion_Rate' and 'Evasion_Rate_Unattacked' instead of 'tdr_attk' and 'tdr'
    metric_columns = [col for col in metric_columns if col not in ['tdr', 'tdr_attk']]
    metric_columns.extend(['Evasion_Rate_Unattacked', 'Evasion_Rate'])

    # Get unique model_types
    model_types = sorted(df['model_type'].unique())

    # Create plots for each metric
    for metric in metric_columns:
        if EXP == 1:
            # Existing logic for EXP == 1 (modified)
            plot_exp1(df, metric, plot_directory, bits_list_per_target)
        elif EXP == 2:
            plot_exp2(df, metric, plot_directory, bits_list_per_target, experiments)
        elif EXP == 3:
            plot_exp3(df, metric, plot_directory, bits_list_per_target, experiments)

    # For EXP == 2 and 3, add the plot for Evasion Rate vs Average L_inf Perturbation
    if EXP == 2 or EXP == 3:
        plot_evasion_vs_perturbation(df, bits_list_per_target, plot_directory, experiments, EXP)


def plot_exp1(df, metric, plot_directory, bits_list_per_target):
    # Get model_types
    model_types = sorted(df['model_type'].unique())

    # For each model_type, create a plot
    for model_type in model_types:
        plt.figure(figsize=(10, 8))  # Increased figure size
        data_plotted = False
        ylabel = ''
        title = ''
        filename = ''
        invert_yaxis = False  # Initialize invert_yaxis flag

        # Plot different targets on the same plot
        for target in bits_list_per_target.keys():
            bits_list = bits_list_per_target[target]
            for bits in bits_list:
                subset = df[
                    (df['target'] == target) &
                    (df['experiment'] == 'optimization') &
                    (df['bits'] == bits) &
                    (df['model_type'] == model_type)
                    ]
                if not subset.empty:
                    data_plotted = True
                    if metric == 'tdr':
                        y_values = 1 - subset[metric]
                        ylabel = 'Evasion Rate (Unattacked)'
                        title = 'Evasion Rate (Unattacked) vs Number of Models (k)'
                        filename = f'Evasion_Rate_Unattacked_vs_k_model{model_type}.png'
                    elif metric == 'tdr_attk':
                        y_values = 1 - subset[metric]
                        ylabel = 'Evasion Rate'
                        title = 'Evasion Rate vs Number of Models (k)'
                        filename = f'Evasion_Rate_vs_k_model{model_type}.png'
                    elif metric == 'auroc_unattacked':
                        y_values = subset[metric]
                        ylabel = 'AUROC (Unattacked)'
                        title = 'AUROC (Unattacked) vs Number of Models (k)'
                        filename = f'AUROC_Unattacked_vs_k_model{model_type}.png'
                        invert_yaxis = True
                    elif metric == 'auroc':
                        y_values = subset[metric]
                        ylabel = 'AUROC'
                        title = 'AUROC vs Number of Models (k)'
                        filename = f'AUROC_vs_k_model{model_type}.png'
                        invert_yaxis = True
                    else:
                        y_values = subset[metric]
                        ylabel = metric_to_label(metric)
                        title = f'{metric_to_label(metric)} vs Number of Models (k)'
                        filename = f'{metric}_vs_k_model{model_type}.png'
                        if 'bitwise' in metric or metric == 'ssim':
                            invert_yaxis = True
                    label = f'{target} ({bits} bits)'
                    plt.plot(subset['k'], y_values, marker='o', label=label)
        if data_plotted:
            plt.xlabel('Number of Models (k)', fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.title(title, fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            if invert_yaxis:
                plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_directory, filename), dpi=300)  # Increased DPI
            plt.close()
        else:
            plt.close()
            print(f'No data available for metric {metric}, model_type {model_type}. Skipping plot.')


def plot_exp2(df, metric, plot_directory, bits_list_per_target, experiments):
    bits_list = bits_list_per_target['hidden']
    # Ensure CNN is in the first row and ResNet in the second row
    model_types_ordered = ['cnn', 'resnet']
    combinations = list(product(model_types_ordered, bits_list))  # Order is (model_type, bits)
    num_cols = 3
    num_rows = 2

    # Define styles for experiments and sources
    experiment_linestyles = {
        'optimization': '-',
        'mean': '--',
        'median': '-.',
        'mean_budget_0.25_clamp': (0, (5, 1)),
        'median_budget_0.25_clamp': (0, (1, 1))
    }

    source_markers = {
        'Stable Diffusion': 'o',
        'Midjourney': 's'
    }

    # Create subplots with 2 rows and 3 columns
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 10))
    fig.suptitle(f'{metric_to_label(metric)} vs Number of Models (k)', fontsize=20)
    data_plotted = False
    invert_yaxis = False  # Initialize invert_yaxis flag

    # Initialize lists to collect handles and labels for the legend
    handles_list = []
    labels_list = []

    for idx, (model_type, bits) in enumerate(combinations):
        row = model_types_ordered.index(model_type)  # CNN is row 0, ResNet is row 1
        col = idx % num_cols
        ax = axes[row, col]
        plot_data_exists = False
        ylabel = ''

        # Plot different experiments on the same subplot
        for experiment in experiments:
            # For each 'source'
            for source in ['Stable Diffusion', 'Midjourney']:
                subset = df[
                    (df['target'] == 'hidden') &
                    (df['experiment'] == experiment) &
                    (df['bits'] == bits) &
                    (df['model_type'] == model_type) &
                    (df['source'] == source)
                    ]

                if not subset.empty:
                    data_plotted = True
                    plot_data_exists = True

                    # Determine y_values and labels based on the metric
                    if metric == 'Evasion_Rate_Unattacked':
                        y_values = subset[metric]
                        ylabel = 'Evasion Rate (Unattacked)'
                        invert_yaxis = False
                    elif metric == 'Evasion_Rate':
                        y_values = subset[metric]
                        ylabel = 'Evasion Rate'
                        invert_yaxis = False
                    elif metric == 'auroc_unattacked':
                        y_values = subset[metric]
                        ylabel = 'AUROC (Unattacked)'
                        invert_yaxis = True
                    elif metric == 'auroc':
                        y_values = subset[metric]
                        ylabel = 'AUROC'
                        invert_yaxis = True
                    else:
                        y_values = subset[metric]
                        ylabel = metric_to_label(metric)
                        if 'bitwise' in metric or metric == 'ssim':
                            invert_yaxis = True
                        else:
                            invert_yaxis = False

                    linestyle = experiment_linestyles.get(experiment, '-')
                    marker = source_markers.get(source, 'o')
                    label = f'{experiment} ({source})'

                    # Plot the data
                    line, = ax.plot(subset['k'], y_values, marker=marker, linestyle=linestyle, label=label)

                    # Collect the handle and label for the legend
                    handles_list.append(line)
                    labels_list.append(label)

        if plot_data_exists:
            ax.set_title(f'Bits: {bits}, Model: {model_type.upper()}', fontsize=16)
            ax.set_xlabel('Number of Models (k)', fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if invert_yaxis:
                ax.invert_yaxis()
        else:
            ax.axis('off')
            print(f'No data available for metric {metric}, bits {bits}, model_type {model_type}. Skipping subplot.')

    # Create a unified legend outside the subplots
    if data_plotted:
        # Remove duplicate labels and handles
        handles_labels = dict(zip(labels_list, handles_list))
        labels_unique = list(handles_labels.keys())
        handles_unique = list(handles_labels.values())

        # Place the legend outside the plot
        fig.legend(handles_unique, labels_unique, loc='center right', fontsize=10, borderaxespad=0.1)
        plt.subplots_adjust(right=0.85)  # Adjust the right boundary to make room for the legend

        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # Adjust layout to make room for the suptitle and legend
        filename = f'{metric}_vs_k_all_bits_models.png'
        plt.savefig(os.path.join(plot_directory, filename))
        plt.close()
    else:
        plt.close()
        print(f'No data available for metric {metric}. Skipping plot.')


def plot_exp3(df, metric, plot_directory, bits_list_per_target, experiments):
    bits_list = bits_list_per_target['hidden']
    model_types_ordered = ['cnn', 'resnet']
    combinations = list(product(model_types_ordered, bits_list))
    num_cols = 3
    num_rows = 2

    # Define styles for experiments
    experiment_colors = {
        'mean_model': 'C1',
        'mean_budget_0.25_clamp_model': 'C2'
    }
    experiment_labels = {
        'mean_model': 'Unnormalized',
        'mean_budget_0.25_clamp_model': 'Clamp'
    }

    # Loop over sources to plot separately
    for source in ['Stable Diffusion', 'Midjourney']:
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 10))
        fig.suptitle(f'{metric_to_label(metric)} vs Number of Models (k) - {source}', fontsize=20)
        data_plotted = False

        for idx, (model_type, bits) in enumerate(combinations):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
            plot_data_exists = False
            ylabel = ''
            invert_yaxis = False  # Initialize invert_yaxis flag

            # Plot 'optimization' data
            subset_opt = df[
                (df['target'] == 'hidden') &
                (df['experiment'] == 'optimization') &
                (df['bits'] == bits) &
                (df['model_type'] == model_type) &
                (df['source'] == source)
                ]

            if not subset_opt.empty:
                plot_data_exists = True
                data_plotted = True
                if metric == 'Evasion_Rate_Unattacked':
                    y_values = subset_opt[metric]
                    ylabel = 'Evasion Rate (Unattacked)'
                    invert_yaxis = False
                elif metric == 'Evasion_Rate':
                    y_values = subset_opt[metric]
                    ylabel = 'Evasion Rate'
                    invert_yaxis = False
                elif metric == 'auroc_unattacked':
                    y_values = subset_opt[metric]
                    ylabel = 'AUROC (Unattacked)'
                    invert_yaxis = True
                elif metric == 'auroc':
                    y_values = subset_opt[metric]
                    ylabel = 'AUROC'
                    invert_yaxis = True
                else:
                    y_values = subset_opt[metric]
                    ylabel = metric_to_label(metric)
                    if 'bitwise' in metric or metric == 'ssim':
                        invert_yaxis = True
                    else:
                        invert_yaxis = False

                label = 'Optimization'
                ax.plot(subset_opt['k'], y_values, marker='o', linestyle='-', label=label)

            # Plot shaded area between min and max values for different model indices
            experiment_bases = ['mean_model', 'mean_budget_0.25_clamp_model']
            for exp_base in experiment_bases:
                # Collect data across model indices
                exp_data = []
                for i in range(2, 12):
                    exp_name = f'{exp_base}_{i}'
                    subset_exp = df[
                        (df['target'] == 'hidden') &
                        (df['experiment'] == exp_name) &
                        (df['bits'] == bits) &
                        (df['model_type'] == model_type) &
                        (df['source'] == source)
                        ]
                    if not subset_exp.empty:
                        plot_data_exists = True
                        data_plotted = True
                        y_value = subset_exp.iloc[0][metric]
                        exp_data.append(y_value)

                if exp_data:
                    y_max = max(exp_data)
                    y_min = min(exp_data)
                    y_avg = sum(exp_data) / len(exp_data)
                    color = experiment_colors[exp_base]
                    label = experiment_labels[exp_base]

                    # Shade the area between y_min and y_max across the entire x-axis
                    ax.axhspan(ymin=y_min, ymax=y_max, facecolor=color, alpha=0.3, label=f'{label} Range')

                    # Plot the average as a horizontal line
                    ax.axhline(y=y_avg, color=color, linestyle='-', label=f'{label} Avg')

            if plot_data_exists:
                ax.set_title(f'Bits: {bits}, Model: {model_type.upper()}', fontsize=16)
                ax.set_xlabel('Number of Models (k)', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=12)
                if invert_yaxis:
                    ax.invert_yaxis()
            else:
                ax.axis('off')
                print(
                    f'No data available for metric {metric}, bits {bits}, model_type {model_type}, source {source}. Skipping subplot.')

        # Create a unified legend outside the subplots
        if data_plotted:
            # Collect handles and labels for legend
            handles, labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    h, l = ax.get_legend_handles_labels()
                    handles.extend(h)
                    labels.extend(l)
            # Remove duplicates
            legend_dict = dict(zip(labels, handles))
            labels_unique = list(legend_dict.keys())
            handles_unique = list(legend_dict.values())

            fig.legend(handles_unique, labels_unique, loc='center right', fontsize=10, borderaxespad=0.1)
            plt.subplots_adjust(right=0.85)
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

            filename = f'{metric}_vs_k_EXP3_{source.replace(" ", "_")}.png'
            plt.savefig(os.path.join(plot_directory, filename))
            plt.close()
        else:
            plt.close()
            print(f'No data available for metric {metric}, source {source}. Skipping plot.')


def plot_evasion_vs_perturbation(df, bits_list_per_target, plot_directory, experiments, EXP):
    bits_list = bits_list_per_target['hidden']
    model_types_ordered = ['cnn', 'resnet']
    combinations = list(product(model_types_ordered, bits_list))
    num_cols = 3
    num_rows = 2

    if EXP == 2:
        # Define styles for experiments and sources
        experiment_linestyles = {
            'optimization': '-',
            'mean': '--',
            'median': '-.',
            'mean_budget_0.25_clamp': (0, (5, 1)),
            'median_budget_0.25_clamp': (0, (1, 1))
        }

        source_markers = {
            'Stable Diffusion': 'o',
            'Midjourney': 's'
        }

        # Loop over sources to plot separately
        for source in ['Stable Diffusion', 'Midjourney']:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 10))
            fig.suptitle(f'Average $L_\\infty$ Perturbation vs Evasion Rate - {source}', fontsize=20)
            data_plotted = False

            # Initialize lists to collect handles and labels for the legend
            handles_list = []
            labels_list = []

            for idx, (model_type, bits) in enumerate(combinations):
                row = model_types_ordered.index(model_type)  # CNN is row 0, ResNet is row 1
                col = idx % num_cols
                ax = axes[row, col]
                plot_data_exists = False

                # Plot different experiments on the same subplot
                for experiment in experiments:
                    subset = df[
                        (df['target'] == 'hidden') &
                        (df['experiment'] == experiment) &
                        (df['bits'] == bits) &
                        (df['model_type'] == model_type) &
                        (df['source'] == source)
                    ]

                    if not subset.empty:
                        data_plotted = True
                        plot_data_exists = True

                        x_values = subset['Evasion_Rate']
                        y_values = subset['noise_L_infinity']

                        linestyle = experiment_linestyles.get(experiment, '-')
                        marker = source_markers.get(source, 'o')
                        label = f'{experiment}'

                        # Plot the data
                        line, = ax.plot(x_values, y_values, marker=marker, linestyle=linestyle, label=label)

                        # Collect the handle and label for the legend
                        handles_list.append(line)
                        labels_list.append(label)

                if plot_data_exists:
                    ax.set_title(f'Bits: {bits}, Model: {model_type.upper()}', fontsize=16)
                    ax.set_xlabel('Evasion Rate', fontsize=14)
                    ax.set_ylabel('Average $L_\\infty$ Perturbation', fontsize=14)
                    ax.grid(True)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                else:
                    ax.axis('off')
                    print(f'No data available for bits {bits}, model_type {model_type}, source {source}. Skipping subplot.')

            # Create a unified legend outside the subplots
            if data_plotted:
                # Remove duplicate labels and handles
                handles_labels = dict(zip(labels_list, handles_list))
                labels_unique = list(handles_labels.keys())
                handles_unique = list(handles_labels.values())

                # Place the legend outside the plot
                fig.legend(handles_unique, labels_unique, loc='center right', fontsize=10, borderaxespad=0.1)
                plt.subplots_adjust(right=0.85)  # Adjust the right boundary to make room for the legend

                plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # Adjust layout to make room for the suptitle and legend
                filename = f'Avg_Linf_vs_Evasion_Rate_EXP2_{source.replace(" ", "_")}.png'
                plt.savefig(os.path.join(plot_directory, filename))
                plt.close()
            else:
                plt.close()
                print(f'No data available for Evasion Rate vs Average L_inf Perturbation for source {source}. Skipping plot.')

    elif EXP == 3:
        # Define styles for experiments
        experiment_colors = {
            'mean_model': 'C1',
            'mean_budget_0.25_clamp_model': 'C2'
        }
        experiment_labels = {
            'mean_model': 'Unnormalized',
            'mean_budget_0.25_clamp_model': 'Clamp'
        }

        # Loop over sources to plot separately
        for source in ['Stable Diffusion', 'Midjourney']:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 10))
            fig.suptitle(f'Average $L_\\infty$ Perturbation vs Evasion Rate - {source}', fontsize=20)
            data_plotted = False

            for idx, (model_type, bits) in enumerate(combinations):
                row = idx // num_cols
                col = idx % num_cols
                ax = axes[row, col]
                plot_data_exists = False

                # Plot 'optimization' data
                subset_opt = df[
                    (df['target'] == 'hidden') &
                    (df['experiment'] == 'optimization') &
                    (df['bits'] == bits) &
                    (df['model_type'] == model_type) &
                    (df['source'] == source)
                ]

                if not subset_opt.empty:
                    plot_data_exists = True
                    data_plotted = True
                    x_values = subset_opt['Evasion_Rate']
                    y_values = subset_opt['noise_L_infinity']
                    label = 'Optimization'
                    ax.plot(x_values, y_values, marker='o', linestyle='-', label=label)

                # Plot scatter points for different model indices
                experiment_bases = ['mean_model', 'mean_budget_0.25_clamp_model']
                for exp_base in experiment_bases:
                    exp_data_x = []
                    exp_data_y = []
                    for i in range(2, 12):
                        exp_name = f'{exp_base}_{i}'
                        subset_exp = df[
                            (df['target'] == 'hidden') &
                            (df['experiment'] == exp_name) &
                            (df['bits'] == bits) &
                            (df['model_type'] == model_type) &
                            (df['source'] == source)
                        ]
                        if not subset_exp.empty:
                            plot_data_exists = True
                            data_plotted = True
                            x_value = subset_exp.iloc[0]['Evasion_Rate']
                            y_value = subset_exp.iloc[0]['noise_L_infinity']
                            exp_data_x.append(x_value)
                            exp_data_y.append(y_value)
                    if exp_data_x and exp_data_y:
                        color = experiment_colors[exp_base]
                        label = experiment_labels[exp_base]
                        ax.scatter(exp_data_x, exp_data_y, color=color, marker='o', label=label)

                if plot_data_exists:
                    ax.set_title(f'Bits: {bits}, Model: {model_type.upper()}', fontsize=16)
                    ax.set_xlabel('Evasion Rate', fontsize=14)
                    ax.set_ylabel('Average $L_\\infty$ Perturbation', fontsize=14)
                    ax.grid(True)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                else:
                    ax.axis('off')
                    print(f'No data available for bits {bits}, model_type {model_type}, source {source}. Skipping subplot.')

            # Create a unified legend outside the subplots
            if data_plotted:
                handles, labels = [], []
                for ax_row in axes:
                    for ax in ax_row:
                        h, l = ax.get_legend_handles_labels()
                        handles.extend(h)
                        labels.extend(l)
                # Remove duplicates
                legend_dict = dict(zip(labels, handles))
                labels_unique = list(legend_dict.keys())
                handles_unique = list(legend_dict.values())

                fig.legend(handles_unique, labels_unique, loc='center right', fontsize=10, borderaxespad=0.1)
                plt.subplots_adjust(right=0.85)
                plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

                filename = f'Avg_Linf_vs_Evasion_Rate_Attacked_EXP3_{source.replace(" ", "_")}.png'
                plt.savefig(os.path.join(plot_directory, filename))
                plt.close()
            else:
                plt.close()
                print(f'No data available for Evasion Rate vs Average L_inf Perturbation for source {source}. Skipping plot.')


if __name__ == '__main__':
    main()

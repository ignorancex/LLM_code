import os
import pandas as pd
from plot_read_logs import get_experiment_settings
from plot_funcs import plot_exp15, plot_exp2, plot_exp3, plot_evasion_vs_perturbation, plot_exp4


def main():
    EXP = int(input('Enter the experiment number (1, 2, 3, 4 or 5): '))
    # EXP = 3
    # Get experiment settings
    targets, plot_directory, experiments, ks, model_types, bits_list_per_target = get_experiment_settings(EXP)

    # Load the processed DataFrame
    df = pd.read_csv(f'processed_data_exp{EXP}.csv')

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

    # Create plots for each metric
    for metric in metric_columns:
        if EXP == 1 or EXP == 5:
            plot_exp15(df, metric, plot_directory, bits_list_per_target, EXP=EXP)
        elif EXP == 2:
            plot_exp2(df, metric, plot_directory, bits_list_per_target, experiments)
        elif EXP == 3:
            plot_exp3(df, metric, plot_directory, bits_list_per_target, experiments)
        elif EXP == 4:
            plot_exp4(df, metric, plot_directory, bits_list_per_target)
        else:
            print(f'Invalid experiment number: {EXP}')
            return

    # For EXP == 2 and 3, add the plot for Evasion Rate vs Average L_inf Perturbation
    if EXP == 2:
        plot_evasion_vs_perturbation(df, bits_list_per_target, plot_directory, experiments, EXP)


if __name__ == '__main__':
    main()

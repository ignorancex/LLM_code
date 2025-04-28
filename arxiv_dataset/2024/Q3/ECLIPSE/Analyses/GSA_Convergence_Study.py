import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import os
import glob
from math import ceil

def read_data(perturbation, patientId, perturbation_dir):
    
    sample_sizes = []

    file_paths = glob.glob(os.path.join(perturbation_dir, f'*_perturb_{perturbation}%_patient_{patientId}.csv'))
    if not file_paths:
        raise FileNotFoundError("No CSV files found in the specified directory.")

    # Get the parameters and output quantities from the first file
    first_file = pd.read_csv(file_paths[0], index_col=0)
    parameters = first_file.columns.tolist()
    output_quantities = first_file.index.tolist()

    data_ST = {output: {param: [] for param in parameters} for output in output_quantities}
    data_SDiff = {output: {param: [] for param in parameters} for output in output_quantities}

    for file_path in sorted(file_paths, key=lambda x: int(x.split('_')[3])):
        file_name = os.path.basename(file_path)
        sample_size = int(file_name.split('_')[1])
        sample_sizes.append(sample_size)
        
        df = pd.read_csv(file_path, index_col=0)
        if 'ST' in file_name:
            for output in output_quantities:
                for param in parameters:
                    data_ST[output][param].append(df.loc[output, param])
        elif 'SDiff' in file_name:
            for output in output_quantities:
                for param in parameters:
                    data_SDiff[output][param].append(df.loc[output, param])

    return data_ST, data_SDiff, sorted(set(sample_sizes)), output_quantities, parameters

def plot_data(data_ST, data_SDiff, sample_sizes, output_quantities, parameters, patientId, perturbation, perturbation_dir):

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10

    cm = 1 / 2.54
    fig_width = 19 * cm
    fig_height = 23 * cm
    num_columns = 2
    num_rows = ceil(len(output_quantities) / num_columns)

    fig_ST, axs_ST = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
    fig_SDiff, axs_SDiff = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))

    C1 = matplotlib.colormaps['tab20'].colors
    C2 = matplotlib.colormaps['Dark2'].colors

    colors = C1 + C2

    axs_ST = axs_ST.flatten()
    axs_SDiff = axs_SDiff.flatten()

    for i, output in enumerate(output_quantities):

        ax_ST = axs_ST[i]
        ax_SDiff = axs_SDiff[i]

        for j, param in enumerate(parameters):
            color = colors[j % len(colors)] 
            ax_ST.plot(sample_sizes, data_ST[output][param], linewidth=1.0, label=param, color=color)
            ax_SDiff.plot(sample_sizes, data_SDiff[output][param], linewidth=1.0, label=param, color=color)

        ax_ST.set_title(f"{output}")
        ax_SDiff.set_title(f"{output}")

        ax_ST.set_xscale('log', base=2)
        ax_SDiff.set_xscale('log', base=2)

        max_power = np.round(np.log2(max(sample_sizes)),0).astype(int)
        min_power = np.round(np.log2(min(sample_sizes)),0).astype(int)
        xticks = [2**x for x in range(min_power - 1, max_power + 1, 2)]

        ax_ST.set_xticks(xticks)
        ax_SDiff.set_xticks(xticks)      

        # Set labels and format axes
        if i // num_columns == num_rows - 1 or i // num_columns == num_rows - 2 and i % num_columns == 1:
            ax_ST.set_xlabel("# of samples")
            ax_SDiff.set_xlabel("# of samples")
        else:
            ax_ST.set_xticklabels([])
            ax_SDiff.set_xticklabels([])

        if i % num_columns == 0:  # Only for the first column
            ax_ST.set_ylabel("Total Order $S_T$")
            ax_SDiff.set_ylabel("$S_T - S_1$")

        ax_ST.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_SDiff.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Disable unused subplots
    for ax in axs_ST[len(output_quantities):]:
        ax.set_visible(False)
    for ax in axs_SDiff[len(output_quantities):]:
        ax.set_visible(False)

    fig_ST.subplots_adjust(top=0.97, bottom=0.055, left=0.07, right=0.85, hspace=0.3, wspace=0.2)
    fig_SDiff.subplots_adjust(top=0.97, bottom=0.055, left=0.07, right=0.85, hspace=0.3, wspace=0.2)

    handles, labels = axs_ST[0].get_legend_handles_labels()
    fig_ST.legend(handles, labels, fontsize=10, bbox_to_anchor=(.8575, .9775), loc="upper left")
    fig_SDiff.legend(handles, labels, fontsize=10, bbox_to_anchor=(.8575, .9775), loc="upper left")

    file_path = os.path.join(perturbation_dir, 'Result_Convergence_Study_perturb_' + 
                   str(perturbation) + '_patient_' + str(patientId)) 
    fig_ST.savefig(file_path + '.pdf')
    fig_ST.savefig(file_path + '.tiff')
    fig_ST.savefig(file_path + '.jpg')

if __name__ == '__main__':

    patientId = 1
    perturbation = 25

    base_path = os.path.join('Sensitivity', 'Results')
    perturbation_dir = os.path.join(base_path, str(perturbation) + '%', '01_Convergence_Analysis')

    data_ST, data_SDiff, sample_sizes, output_quantities, parameters = read_data(perturbation, patientId, perturbation_dir)
    plot_data(data_ST, data_SDiff, sample_sizes, output_quantities, parameters, patientId, perturbation, perturbation_dir)
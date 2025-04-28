import re
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score
import seaborn as sns
import string
from copy import deepcopy

def read_patient_txt(file_path):
    """
    Read patient data from a text file and extract the outputs.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of dictionaries containing the extracted outputs. Each dictionary
              has the following keys: 'Parameter', 'Simulated', 'Measured'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """

    with open(file_path, 'r') as file:
        content = file.read()

    outputs_section = re.search(r'Outputs ###########(.+?)######### Parameters', content, re.DOTALL).group(1)

    outputs = []
    for line in outputs_section.strip().split('\n')[1:]:  # Skipping the header line
        parts = line.split()
        if len(parts) == 3:  # Ensure there are exactly 3 parts: parameter, simulated, measured
            param, sim, meas = parts
            outputs.append({
                'Parameter': param,
                'Simulated': float(sim),
                'Measured': float(meas)
            })

    return outputs

def get_optimization_results(file_paths):
    """
    Retrieves optimization results from the specified directory.

    Args:
        file_paths (list): A list of file paths containing the optimization results.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from all the result files.
    
    Raises:
        FileNotFoundError: If no result files for optimization are found in the specified directory.
    """

    if not file_paths:
        raise FileNotFoundError("No result files for optimization found in the specified directory.")

    all_patients_data = []

    for file_path in file_paths:
        patient_id = re.search(r'result_optimization_patient_([0-9]+)\.txt', file_path).group(1)
        patient_data = read_patient_txt(file_path)
        for data_point in patient_data:
            data_point['Patient'] = f'Patient_{patient_id}'
            all_patients_data.append(data_point)

    return pd.DataFrame(all_patients_data)

def scatter_plot(scatter_data, categories, gs_parent, fig_parent, subplot_labels):
    """
    Generate a scatter plot for the given scatter data.

    Args:
        - scatter_data (DataFrame): The scatter data containing the simulated and measured values.
        - categories (dict): A dictionary containing the category information for the scatter plot.
                       Each category should have a 'keys' list containing the parameter keys and a 'unit' string.
        - gs_parent (GridSpec): The parent GridSpec object for the scatter plot.
        - fig_parent (Figure): The parent Figure object for the scatter plot.
        - subplot_labels (list): A list of subplot labels for the scatter plot.

    Returns:
        - color_map (dict): A dictionary containing the color mapping for the scatter plot.
    Raises:
        -
    """

    unique_parameters = sum((category_info['keys'] for category_info in categories.values()), [])
    palette = sns.color_palette("colorblind", len(unique_parameters))  # You can choose any appropriate palette
    color_map = dict(zip(unique_parameters, palette))

    gs_scatter = GridSpec.GridSpecFromSubplotSpec(1, len(subplot_labels), subplot_spec=gs_parent)

    legend_handles = []

    ii = 0
    for category_name, category_info in categories.items():
        ax = plt.subplot(gs_scatter[ii])
        category_keys = category_info['keys']
        unit = category_info['unit']
        all_simulated = []
        all_measured = []

        category_data = scatter_data[scatter_data['Parameter'].isin(category_keys)]
        
        if not category_data.empty:
            for parameter in category_keys:
                parameter_data = category_data[category_data['Parameter'] == parameter]
                if not parameter_data.empty:        
                    scatter = ax.scatter(parameter_data['Simulated'], parameter_data['Measured'], 
                            label=parameter, color=color_map.get(parameter, '#000000'),
                            edgecolors='black', linewidths=0.5, s=25)
                    legend_handles.append(scatter)
                    all_simulated.append(parameter_data['Simulated'])
                    all_measured.append(parameter_data['Measured'])

            if all_simulated and all_measured:

                all_simulated = np.concatenate(all_simulated)
                all_measured = np.concatenate(all_measured)

                r_squared = r2_score(all_measured, all_simulated)

                ax.text(0.05, 0.98, f'R² = {r_squared:.2f}', transform=ax.transAxes, verticalalignment='top')

            # Plot 1:1 line
            max_val = np.max([category_data['Simulated'].max(), category_data['Measured'].max()])
            ax.plot([0, max_val], [0, max_val], 'k--')

            ax.set_xlabel(f'Simulated {category_name.lower()} in {unit}')
            ax.set_ylabel(f'Measured {category_name.lower()} in {unit}')

            # Create subplot labeling
            col_frac = ii / len(subplot_labels)
            if ii == 1 or ii == 2: col_frac -= 0.01
            transform = transforms.blended_transform_factory(
                                    fig_parent.transFigure, ax.transData)
            ax.text(col_frac + 0.04, ax.get_ylim()[1] + 0.0 * np.diff(ax.get_ylim()), 
                subplot_labels[ii], transform=transform, fontweight='bold', verticalalignment='top', horizontalalignment='left')

            ii += 1

    fig_parent.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(.525, .485), \
                      ncol=len(scatter_data['Parameter'].unique()), frameon=False, handletextpad=.1, scatteryoffsets=[0.4])

    return color_map

def box_plot(scatter_data, categories, gs_parent, fig_parent, subplot_labels, color_map):
    """
    Generate a box plot for scatter data based on different categories.

    Args:
        - scatter_data (DataFrame): The scatter data containing the parameters and values.
        - categories (dict): A dictionary containing the category names, keys, and units.
        - gs_parent (GridSpec): The parent GridSpec object for the box plot.
        - fig_parent (Figure): The parent Figure object for the box plot.
        - subplot_labels (list): A list of subplot labels for the box plot.
        - color_map (dict): A dictionary containing the color mapping for the box plot.

    Returns:
        -

    Raises:
        -
    """

    num_parameters_per_group = {category_name: len([key for key in category_info['keys'] if key in scatter_data['Parameter'].unique()])
                                for category_name, category_info in categories.items()}
    width_ratios = [num_parameters_per_group[category_name] for category_name in categories.keys() if num_parameters_per_group[category_name] > 0]
    total_width = sum(width_ratios)
    cum_widths = np.array(np.cumsum([0] + width_ratios[:-1]) / total_width)
    cum_widths[1] -= 0.0425

    gs_box = GridSpec.GridSpecFromSubplotSpec(1, len(width_ratios), subplot_spec=gs_parent, width_ratios=width_ratios)

    ii = 0
    for category_name, category_info in categories.items():    
        ax = plt.subplot(gs_box[ii])
        category_keys = category_info['keys']
        category_data = scatter_data[scatter_data['Parameter'].isin(category_keys)]

        if not category_data.empty:
  
            data_to_plot = []
            labels = []
            box_width = 0.8
            scatter_xs = []
            scatter_labels = []

            index = 0
            for key in category_keys:
                sim_data = category_data[category_data['Parameter'] == key]['Simulated'].dropna().values
                meas_data = category_data[category_data['Parameter'] == key]['Measured'].dropna().values
                if len(sim_data) > 0 and len(meas_data) > 0:
                    data_to_plot.extend([sim_data, meas_data])
                    labels.append(key)

                    scatter_xs.extend(np.random.normal(index * 2 + 1, 0.1, sim_data.shape[0]))
                    scatter_xs.extend(np.random.normal(index * 2 + 2, 0.1, meas_data.shape[0]))
                    scatter_labels.extend([key] * len(sim_data))
                    scatter_labels.extend([key] * len(meas_data))

                    index+=1
                    
            bp = ax.boxplot(data_to_plot, positions=np.arange(1, 2 * len(labels) + 1), widths=box_width, patch_artist=True,
                            whis=(0, 100), zorder=1)

            for pos, val, label in zip(scatter_xs, np.concatenate(data_to_plot), scatter_labels):
                color = color_map.get(label, '#000000')
                colorTransparent = color + (0.6,)
                ax.scatter(pos, val, facecolor=colorTransparent,
                           edgecolors='black', linewidths=0.5, zorder=2, s=25)
            
            # Set colors for boxes
            for i, patch in enumerate(bp['boxes']): 
                param_index = i // 2  # find parameter index based on box index
                key = labels[param_index]
                # Set color and transparency of boxes
                patch.set_facecolor(color_map.get(key, '#000000'))
                #r, g, b, a = patch.get_facecolor()
                #pastel_color = ((r + 1) / 2, (g + 1) / 2, (b + 1) / 2, a)
                #patch.set_facecolor(pastel_color)

                #  Set hatching of boxes of simulated data
                patch.set_hatch('///' if i % 2 == 0 else '')
                patch.set_edgecolor((0, 0, 0, 1.)) # TODO: Check this here!

            ax.set_xticks(np.arange(1.5, 2 * len(labels) + 0.5, 2))
            ax.set_xticklabels(labels)
            ax.set_ylabel(f'{category_name} in {category_info["unit"]}')
            ax.set_xlabel('')
            ax.set_ylim(bottom=0)
            ax.grid(False)

            # Create subplot labeling
            transform = transforms.blended_transform_factory(
                                    fig_parent.transFigure, ax.transData)

            ax.text(cum_widths[ii] + 0.04, ax.get_ylim()[1] + 0.0 * np.diff(ax.get_ylim()), 
                subplot_labels[ii], transform=transform, fontweight='bold', verticalalignment='top', horizontalalignment='left')

            ii += 1
    
    # Adding legend for simulated and measured data
    simulated_patch = mpatches.Patch(edgecolor='black', facecolor='white', label='Simulated', hatch='///')
    measured_patch = mpatches.Patch(edgecolor='black', facecolor='white', label='Measured')
    fig_parent.legend(handles=[simulated_patch, measured_patch], loc='lower center', ncol=ii, frameon=False, bbox_to_anchor=(0.25, 0.0))

def combined_plot(scatter_data, categories, directory_path):

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10
    cm = 1/2.54
    fig_width = 19*cm
    fig_height = 23*cm

    fig_parent = plt.figure(figsize=(fig_width, fig_height))
    gs_parent = GridSpec.GridSpec(2, 1, figure=fig_parent)

    # Create subplot labels
    num_plots = sum(any(key in scatter_data['Parameter'].unique() for key in category_info['keys']) for category_info in categories.values())
    subplot_labels = [f"{letter})" for letter in string.ascii_lowercase[:2*num_plots]]
    scatter_labels = subplot_labels[:num_plots]
    box_labels = subplot_labels[num_plots:]

    color_map = scatter_plot(scatter_data, categories, gs_parent[0], fig_parent, scatter_labels)
    box_plot(scatter_data, categories, gs_parent[1], fig_parent, box_labels, color_map)

    gs_parent.tight_layout(figure=fig_parent, h_pad=4.0, rect=[0, 0.03, 1, 1])

    file_path = os.path.join('Analyses', 'Results', '03_Fitting')

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = 'Optimization_results_scatter_box_plot'

    fig_parent.savefig(os.path.join(file_path, file_name) + ".pdf")
    fig_parent.savefig(os.path.join(file_path, file_name) + ".svg")
    fig_parent.savefig(os.path.join(file_path, file_name) + ".png")   

if __name__ == '__main__':

    # Get optimization results from all patient txt files
    directory_path = os.path.join('Optimization', 'Results', 'patients1-30', 'First_Round')
    file_paths = glob.glob(os.path.join(directory_path, 'result_optimization_patient_*.txt'))

    scatter_data = get_optimization_results(file_paths)

    # Splitting the data into pressure and flow based on parameter
    categories = {
        'Volume': {
            'keys': ['ESVLV', 'EDVLV', 'ESVLA', 'EDVLA', 'ESVRV', 'EDVRV', 'ESVRA', 'EDVRA'],
            'unit': 'mL'
        },
        'Pressure': {
            'keys': ['SP', 'DP', 'MAP', 'SPAP', 'DPAP', 'MPAP', 'PCWP'],
            'unit': 'mmHg'
        },
        'Flow': {
            'keys': ['CO', 'PF'],
            'unit': 'L/min'
        }
    }

    combined_plot(scatter_data, categories, directory_path)
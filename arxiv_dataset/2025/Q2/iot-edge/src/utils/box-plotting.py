import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import Circle, RegularPolygon, Patch
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors
from datetime import datetime

distributions = ['k0s', 'k3s', 'k8s', 'kubeEdge', 'openYurt']
distValues = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

# Create the dictionary
mapping = dict(zip(distributions, distValues))

def scatter_plots_with_trend_lines(all_data, title, xlabel, ylabel, toSave=False):
    combined_data = pd.concat(all_data, ignore_index=True)
    hosts = combined_data['hostname'].unique()

    for host in hosts:
        plt.figure(figsize=(15, 8))
        host_data = combined_data[combined_data['hostname'] == host]
        
        # Plot scatter with trend lines for each distribution
        for dist in host_data['dist'].unique():
            dist_data = host_data[host_data['dist'] == dist]
            sns.regplot(data=dist_data, x='minutes', y='value', scatter_kws={'s': 20, 'alpha': 0.5}, line_kws={'label': dist}, ci=None)

        plt.title(f'{title} for {host}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title='Distribution')

        if toSave:
            snake_title = title.replace(' ', '_').lower()
            file_name = f'{snake_title}_for_{host}'
            plt.savefig(file_name, format='pdf')
            # plt.savefig(file_name)
        else:
            plt.show()

# TODO: remove outliers and check
def box_plotting(all_data, title, xlabel, ylabel, toSave=False, unit='', showfliers=False):
    """
    Create and save or display box plots for given data files.

    Args:
    all_data (list): A list of dataframes.
    title (str): The title for the plots.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    toSave (bool): If True, save the plots instead of displaying them.
    """
    combined_data = pd.concat(all_data, ignore_index=True)
    hosts = combined_data['hostname'].unique()

    for host in hosts:
        host_data = combined_data[combined_data['hostname'] == host]
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=host_data, x='minutes', y='value', hue='dist', showfliers=showfliers)
        plt.title(f'{title} for {host}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title='Distribution')

        if toSave:
            snake_title = title.replace(' ', '_').lower()
            file_name = f'{snake_title}_for_{host}.pdf'
            sub_dir = 'box' if showfliers == True else 'box-without-fliers'
            plt.savefig(f'../diagrams/results/{sub_dir}/{unit}-{file_name}', format='pdf')
        else:
            plt.show()

def line_plotting(all_data, title, xlabel, ylabel, toSave=False, unit=''):
    """
    Create and save or display line plots for given data files.

    Args:
    all_data (list): A list of dataframes.
    title (str): The title for the plots.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    toSave (bool): If True, save the plots instead of displaying them.
    """
    combined_data = pd.concat(all_data, ignore_index=True)
    hosts = combined_data['hostname'].unique()

    for host in hosts:
        host_data = combined_data[combined_data['hostname'] == host]
        plt.figure(figsize=(6, 4.1))
        colors = ['b', 'r', 'g', 'm', 'y']
        sns.lineplot(data=host_data, x='minutes', y='value', hue='dist', palette=colors)
        plt.title(f'{title} for {host}', fontsize='x-large')
        plt.xlabel(xlabel, fontsize='x-large')
        plt.ylabel(ylabel, fontsize='x-large')
        # uncomment this for plotting network outage and recovery time - start
        # plt.figure(figsize=(8, 4.1))
        # if len(all_data[0]['outage_start']) > 0:
        #     # two lines below are for marking network outage
        #     plt.axvline(x = all_data[0]['outage_start'][0], color = 'r', label = 'outage start', linestyle=':')
        #     plt.axvline(x = all_data[0]['outage_end'][0], color = 'r', label = 'outage end', linestyle='dashdot')
            
        #     colors = ['b', 'r', 'g', 'm', 'y']
        #     for i, dist in enumerate(host_data['dist'].unique()):
        #         dist_data = host_data[host_data['dist'] == dist]
        #         xPoint = all_data[0]['outage_end'][0] + dist_data['recovery'].mean()
        #         plt.axvline(x=xPoint, color=colors[i], label=f'rec.time of {dist}', alpha=0.5, linestyle='--') # TODO: find out how to plot a dot for intersection point of corresponding distribution
        # uncomment this for plotting network outage and recovery time - end

                # # ymin = dist_data.loc[dist_data['minutes'] + 0.1 <= x, 'value'].iloc[0]
                # ymax = dist_data.loc[dist_data['minutes'] + 0.1 <= x, 'value'].iloc[0]
                # plt.vlines(x = xPoint, ymin = ymin, ymax = ymax, colors = colors[i], label = 'recovery')
                

                # plt.axvline(x = all_data[0]['outage_start'][0] + dist_data['minutes'].quantile(0.98).max(), color = 'y', label = '98th percentile')
                # plt.axvline(x = all_data[0]['outage_start'][0] + dist_data['minutes'].quantile(0.95).max(), color = 'y', label = '95th percentile')
                # plt.axvline(x = all_data[0]['outage_start'][0] + dist_data['minutes'].quantile(0.90).max(), color = 'y', label = '90th percentile')

            # plt.axvline(x = all_data[0]['outage_end'][0]+ host_data['recovery'].mean(), color = 'y', label = 'recovery')

            # fill between two lines for marking network outage
            # max = host_data['value'].quantile(0.98).max()
            # min = host_data['value'].min()
            # plt.fill_betweenx([min, max ], all_data[0]['outage_start'][0], all_data[0]['outage_end'][0], color='red', alpha=0.5)
        plt.legend(title='Distribution')
        # legend = plt.legend(title='Distribution', loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.tight_layout(rect=[0, 0, 1, 1])
        # legend.set_visible(False) # hide the legend
        if toSave:
            snake_title = title.replace(' ', '_').lower()
            file_name = f'{snake_title}_for_{host}'
            plt.savefig(f'../diagrams/results/latest/short/{file_name}-{unit}-with-legend.pdf', format='pdf', bbox_inches='tight')
        else:
            plt.show()

def create_plots(files, title, xlabel, ylabel, toSave=False, plot_type='scatter', uniteWorkers=False, reliabilityWorker=False, reliabilityTest=''):
    """
    Create and save or display box plots for given data files.

    Args:
    files (list): A list of file paths to the data files.
    title (str): The title for the plots.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    toSave (bool): If True, save the plots instead of displaying them.
    """
    all_data = []
    unit = ''
    for file in files:
        path_components = file.split('/')
        dist = path_components[2]
        unit = path_components[-1].split('-')[-1].split('.')[0]
        data = pd.read_csv(file)

        if reliabilityWorker:
            paths = file.split('/')[:-1]
            filePath = '/'.join(paths) + f'/ansible_output_{dist}_{title}-{paths[-1][-1]}.txt'
            opened_file = open(filePath, 'r')
            first_line = opened_file.readline()
            opened_file.close()
            node_number = first_line.split(' ')[-1].split(':')[0]
            data['hostname'] = data['hostname'].apply(lambda x: 'failed' if x == f'node_{int(node_number)+1}' else 'worker' if 'node' in x else x)
        else:
            if uniteWorkers:
                data['hostname'] = data['hostname'].apply(lambda x: 'worker' if 'node' in x else x)

        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')   

        if unit == 'cpu':
            data['value'] = 100 - data['value']

        if unit == 'net':
            # Ensure 'value' is positive for all rows, assuming negative values indicate outgoing data
            data['value'] = data['value'].abs()
            # Group by 'hostname' and 'timestamp' to calculate total traffic (sum of InOctets and OutOctets)
            data_grouped = data.groupby(['hostname', 'timestamp']).agg(TotalTraffic=('value', 'sum')).reset_index()

            # Replace the original 'data' DataFrame with the grouped one
            data = data_grouped
            data['value'] = data['TotalTraffic'] / 1024  # Convert bytes to kilobytes
        

        min_timestamp = data['timestamp'].min()
        data['timestamp'] -= min_timestamp
        data['minutes'] = (data['timestamp'].dt.total_seconds() / 60).round().astype(int)
        data['dist'] = mapping[dist]

        if reliabilityTest != '':
            start_in_min = 250/60 # after start of a test, the script waits for 250 seconds
            duration = 500 if 'long' == reliabilityTest else 100
            wait_time = 200 if 'long' == reliabilityTest else 600
            data['outage_start'] = start_in_min
            data['outage_end'] = start_in_min + duration/60 # duration in minutes


            paths = file.split('/')[:-1]
            filePath = '/'.join(paths) + f'/tmp-recovery.txt' # actually it is not recovery time, it is the time of before and after
            opened_file = open(filePath, 'r')
            durationOfTest = int(opened_file.readline())
            opened_file.close()
            data['recovery'] = ((durationOfTest - 250 - wait_time - duration) / 60) # 250 is the waiting time, 600 is post test waiting time,duration is the duration of the test 

        all_data.append(data)

    if plot_type == 'box':
        box_plotting(all_data, title, xlabel, ylabel, toSave, unit, showfliers=False)
    elif plot_type == 'line':
        line_plotting(all_data, title, xlabel, ylabel, toSave, unit)
    elif plot_type == 'recovery':
        plot_recovery_time(all_data, title, xlabel, ylabel, toSave)
    else:
        scatter_plots_with_trend_lines(all_data, title, xlabel, ylabel, toSave, unit)
    

toSave = True # saved them manually since it is not worth handling them via code
# testCases = ['idle', 'cp_light_1client', 'cp_heavy_8client', 'cp_heavy_12client', 'dp_redis_density', 'reliability-control', 'reliability-control-no-pressure-long', 'reliability-worker', 'reliability-worker-no-pressure-long']
# testCases = ['idle', 'cp_light_1client', 'reliability-control-no-pressure-long', 'reliability-worker', 'reliability-worker-no-pressure-long']
testCases = ['reliability-control', 'reliability-worker']
metrics = ['cpu', 'ram', 'net', 'disk']
uniteWorkers=True
def create_plots_time_series(plot_type='scatter'):
    for unit in metrics:
        for test in testCases:
            files = [] 
            if 'worker' in test:
                reliabilityTestsForWorker = True
            else:
                reliabilityTestsForWorker = False
            for dist in distributions:
                for i in range(2, 5):
                    files.append(f'../k-bench-results/{dist}/{test}/{test}-{i}/{test}-{i}-{unit}.csv')
            ylabel = 'CPU Usage (%)' if unit == 'cpu' else 'Memory Usage (Mb)' if unit == 'ram' else 'Network load (kB)' if unit == 'net' else 'Disk Usage (%)'
            reliabilityTest = 'long' if 'long' in test else 'short' if 'reliability' in test else ''
            create_plots(files, f'{test}', 'Minutes', ylabel, toSave, plot_type, uniteWorkers, reliabilityTestsForWorker, reliabilityTest)

# create_plots_time_series('line')


def plot_recovery_time(all_data, title, xlabel, ylabel, toSave=False):
    """
    Create and save or display box plots for given data files.

    Args:
    all_data (list): A list of dataframes.
    title (str): The title for the plots.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    toSave (bool): If True, save the plots instead of displaying them.
    """
    combined_data = pd.concat(all_data, ignore_index=True)
    hosts = combined_data['hostname'].unique()

    for host in hosts:
        host_data = combined_data[combined_data['hostname'] == host]
        plt.figure(figsize=(8, 4.8))
        sns.boxplot(data=host_data, x='dist', y='recovery', hue='dist', showfliers=False)
        plt.title(f'{title}', fontsize='x-large')
        plt.xlabel(xlabel, fontsize='x-large')
        plt.ylabel(ylabel, fontsize='x-large')
        if (title == 'reliability-worker'):
            # axis y should be placed on the right side
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
        plt.tick_params(axis='y', labelsize='x-large')
        plt.tick_params(axis='x', labelsize='x-large')
        
        plt.grid(True)

        if toSave:
            snake_title = title.replace(' ', '_').lower()
            file_name = f'{snake_title}_for_{host}.pdf'
            plt.savefig(f'../diagrams/recovery/{file_name}', format='pdf')
        else:
            plt.show()

def create_plots_recovery(plot_type='recovery'):
        for test in ['reliability-control', 'reliability-worker']:
            files = [] 
            for unit in metrics:
                if 'worker' in test:
                    reliabilityTestsForWorker = True
                else:
                    reliabilityTestsForWorker = False
                for dist in distributions:
                    for i in range(2, 5):
                        files.append(f'../k-bench-results/{dist}/{test}/{test}-{i}/{test}-{i}-{unit}.csv')
            ylabel = 'Recovery Time (min)'
            reliabilityTest = 'long' if 'long' in test else 'short' if 'reliability' in test else ''
            create_plots(files, f'{test}', 'Distributions', ylabel, toSave, plot_type, uniteWorkers, reliabilityTestsForWorker, reliabilityTest)

create_plots_recovery('recovery')



######## spider plots below
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def return_single_avg_value(files, metric, machine):
    all_data = []
    avg_value = 0
    for file in files:
        data = pd.read_csv(file)
        if metric == 'cpu':
            data['value'] = 100 - data['value']
        if metric == 'net':
            data['value'] = data['value'].abs()
            # Group by 'hostname' and 'timestamp' to calculate total traffic (sum of InOctets and OutOctets)
            data_grouped = data.groupby(['hostname', 'timestamp']).agg(TotalTraffic=('value', 'sum')).reset_index()

            # Replace the original 'data' DataFrame with the grouped one
            data = data_grouped
            data['value'] = data['TotalTraffic'] / 1024  # Convert bytes to kilobytes

        if machine == 'worker':
            data = data[data['hostname'].str.match('^node_.*')].copy()
        else:
            data = data[data['hostname'].str.match('^master$')].copy()       
        all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    avg_value = combined_data['value'].mean()
    if (metric == 'ram'):
        if machine == 'worker':
            avg_value =( avg_value * 100) / 4096
        else:
            avg_value  = (avg_value * 100) / 65336
    # if (metric == 'net'):
    #     avg_value = (avg_value * 100) / 3
    return avg_value
    

def prepare_data_set_for_machine(machine):
    data = dict()
    for test in testCases:
        data[test] = dict()
        for dist in distributions:
            mappedDist = mapping[dist]
            data[test][mappedDist] = []
            for metric in metrics:
                files = []
                for i in range(2, 5):
                    files.append(f'../k-bench-results/{dist}/{test}/{test}-{i}/{test}-{i}-{metric}.csv')
                avg = return_single_avg_value(files, metric, machine)
                # print(f'{test} for {dist} and {metric} is {avg}')
                data[test][mappedDist].append(avg)
    return data




def create_spider_plots_for_test_cases(toSave=False):

    data_set = [prepare_data_set_for_machine('master'), prepare_data_set_for_machine('worker')]
    
    N = len(metrics)
    theta = radar_factory(N, frame='polygon')
    spoke_labels = metrics

    colors = ['b', 'r', 'g', 'm', 'y']
    fig, axs = plt.subplots(figsize=(10, 35), nrows=len(testCases), ncols=2, # each distribution should be presented as two diagrams: master and worker
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.90, bottom=0.05)
    fig.tight_layout(pad=1.7)

    # I want to have plots a bit far from each other

    for p, data in enumerate(data_set):
        for i, (test, dists) in enumerate(data.items()):
            max_value = 0
            for j, (dist, values) in enumerate(dists.items()):
                # if (i >= 2 or j >= 2):
                #     print(f' i = {i} and j = {j}, test = {test} and dist = {dist}')
                if (len(values) == 0):
                    print(f'No data for {test} and {dist}')
                    continue
                # print(values)
                axs[i, p].plot(theta, values, color=colors[j])
                axs[i, p].fill(theta, values, facecolor=colors[j], alpha=0.5)   
                max_value = max(max_value, max(values)) 
            
            axs[i, p].set_varlabels(spoke_labels)
            # bigger font for labels
            axs[i, p].tick_params(axis='x', labelsize='xx-large')
            axs[i, p].tick_params(axis='y', labelsize='xx-large')
            noteType = 'Master' if p == 0 else 'Worker'
            axs[i, p].set_title(f'{test} for {noteType} (%)', weight='bold', size='large', position=(0.5, 1.1))
            axs[i, p].set_ylim(0, math.ceil(max_value))



    handles = []
    for i, dist in enumerate(distributions):
        handles.append(Patch(color=colors[i], label=mapping[dist]))
    legend = axs[0, 0].legend(handles, distValues, loc=(0.9, .80),
                              labelspacing=0.2, fontsize='x-large')

    # Adjust the layout to prevent overlapping of subplots
    # plt.subplots_adjust(hspace=0.5)


    if toSave:
            plt.savefig('../diagrams/spider_diagrams-dp-redis_only_v2.pdf', format='pdf')
    else:
        plt.show()
    
# create_spider_plots_for_test_cases(True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter

distributions = ['k0s', 'k3s', 'k8s', 'kubeEdge', 'openYurt']
values = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

# Create the dictionary
mapping = dict(zip(distributions, values))

# type,k8s,tests,test-numbers,number-of-pods,metics,value
# pod,k0s,cp_heavy_8client,1,16,throughput,1.9339411
# pod,k0s,cp_heavy_8client,1,16,latency,1.9339411
# deployment,k0s,cp_heavy_8client,1,80,throughput,4.9721656
# deployment,k0s,cp_heavy_8client,1,80,latency,4.9721656
# pod,k0s,cp_heavy_8client,2,16,throughput,2.1950681
# pod,k0s,cp_heavy_8client,2,16,latency,2.1950681

def plot_latency_diagram(file_path, test, saveFile=False):

    all_data = []
    data = pd.read_csv(file_path)

    df = pd.DataFrame(data)

    # df = df[df['metics'] == 'throughput']
    # df = df[df['metics'] == 'latency']
    df = df[df['type'] == 'deployment']

    df['value'] = df['value'].astype(float)

    df['k8s'] =  df['k8s'].map(mapping)

    # Prepare DataFrame for plotting by melting it
    df_melted = df.melt(id_vars=["type", "k8s", "number-of-pods", "metics"], 
                        value_vars=["value"],
                        var_name="statistic", value_name="value_name")

    print(df_melted)

    # colors = ['b', 'r', 'g', 'm', 'y']
    # g = sns.FacetGrid(df_melted, col="number-of-pods", height=7, aspect=1, col_wrap=2, sharey=False, sharex=True)
    # g.map_dataframe(sns.boxplot, x="k8s", y="value_name", hue="k8s", showfliers=False, palette=colors)
    # g.add_legend()
    # g.set_titles("{col_name} pods " + f"in {test}")
    # g.set_axis_labels("Distributions", "Throughput (pods/min)")


    # Create a new subset of data by grouping data by the "number-of-pods" column, there are only 2 unique values
    grouped_data = df.groupby('metics')
    combined_data = pd.concat([grouped_data.get_group(gp) for gp in grouped_data.groups], ignore_index=True)
    # devide into two dataframes
    dataset1 = combined_data[combined_data['metics'] == 'latency']
    nPods = dataset1['number-of-pods'].unique()[0]
    print(dataset1)

    boxprops1 = dict(linewidth=3, fill=None, color='red')
    capprops1 = dict(linewidth=3, color='red')
    medianprops1 = dict(linewidth=3, color='red')
    fig, ax1 = plt.subplots(figsize=(6,6))
    whiskerprops1 = dict(color='red')
    sns.boxplot(data=dataset1, x='k8s', y='value', ax=ax1, color='red', whiskerprops=whiskerprops1, showfliers=False, boxprops=boxprops1, capprops=capprops1, medianprops=medianprops1 ) #width=0.5,
    ax1.set_ylabel(f'Deployment Latency of {nPods} pods (ms)', color='red', fontsize='x-large')
    # ax1.set_ylabel(f'Pod Creation Latency of {nPods} pods (ms)', color='red', fontsize='x-large')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel('Distributions', fontsize='x-large')

    dataset2 = combined_data[combined_data['metics'] == 'throughput']
    # nPods = dataset2['metics'].unique()[0]
    boxprops2 = dict(linewidth=3, fill=None, color='blue')
    capprops2 = dict(linewidth=3, color='blue')
    medianprops2 = dict(linewidth=3, color='blue')
    ax2 = ax1.twinx()
    whiskerprops2 = dict(color='blue')
    sns.boxplot(data=dataset2, x='k8s', y='value', ax=ax2, color='blue', whiskerprops=whiskerprops2, showfliers=False, boxprops=boxprops2, width=0.5, capprops=capprops2, medianprops=medianprops2)
    # ax2.set_ylabel(f'Throughput Pod Creation of {nPods} pods(pods/min) ', color='blue', fontsize='x-large')
    ax2.set_ylabel(f'Throughput Deployment of {nPods} pods (pods/min)', color='blue', fontsize='x-large')
    ax2.tick_params(axis='y', labelcolor='blue')
    fig.tight_layout()  
    plt.grid(True)
    plt.tight_layout()

    # Set y-axis step size
    # g.axes[0].yaxis.set_major_locator(plt.MultipleLocator(1))
    # g.axes[0].yaxis.set_minor_locator(plt.MultipleLocator(1000))
    # g.set(yscale='log')

    # Overlay boxplots on bars
    # g.map_dataframe(sns.boxplot, x="type", y="value_name", hue="k8s", showfliers=False, palette=colors)
    # g.ax.set_ylim(bottom=0)
    if saveFile:
        # plt.savefig(f'../../diagrams/throughput-latency-statistics/latency-{test}.pdf', format='pdf')
        plt.savefig(f'../../diagrams/throughput-latency-statistics/throughput-and-latency-{nPods}-{test}.pdf', format='pdf')
    else:
        plt.show()


tests=["cp_light_1client", "cp_heavy_8client", "cp_heavy_12client"]

for test in tests:
    plot_latency_diagram(f"../../k-bench-results/throughput-latency-statistics/{test}.csv",test, saveFile=True)
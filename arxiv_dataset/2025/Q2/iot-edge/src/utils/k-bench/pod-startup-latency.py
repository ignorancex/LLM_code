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

# this is the data format
# k8s,tests,test-numbers,number-of-pods,medians,mins,maxs
# k0s,cp_light_1client,1,2,1000,0,1000
# k0s,cp_light_1client,1,10,1000,1000,1000
# k0s,cp_light_1client,2,2,1000,1000,1000
# k0s,cp_light_1client,2,10,1000,1000,1000
# k0s,cp_light_1client,3,10,1000,1000,1000

distributions = ['k0s', 'k3s', 'k8s', 'kubeEdge', 'openYurt']
values = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

# Create the dictionary
mapping = dict(zip(distributions, values))

def plot_diagram(file_path, saveFile=False):

    all_data = []
    data = pd.read_csv(file_path)

    data['k8s'] = data['k8s'].map(mapping)

    df = pd.DataFrame(data)

    # Prepare DataFrame for plotting by melting it
    df_melted = df.melt(id_vars=["k8s", "number-of-pods"], 
                        value_vars=["medians", "medians", "medians"],
                        var_name="statistic", value_name="value")

    df_melted['value'] = df_melted['value'].astype(float)/1000

    colors = ['b', 'r', 'g', 'm', 'y']
    g = sns.FacetGrid(df_melted, height=3, col="statistic", aspect=2, col_wrap=1, sharey=False, sharex=False)
    g.map_dataframe(sns.lineplot, x="number-of-pods", y="value", hue="k8s", palette=colors,  err_style="bars", errorbar=("se", 2))
    legend = g.add_legend()
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.9), ncol=2)
    g.set_titles("Pods startup latency ({col_name})")
    g.set_axis_labels("Number of Pods", "Latency (sec)")
    if saveFile:
        plt.savefig(f'../../diagrams/pod-startup-latency.pdf', format='pdf')
    else:
        plt.show()


plot_diagram(f"../../k-bench-results/pod-startup-latency/pod-startup-latency.csv", saveFile=True)
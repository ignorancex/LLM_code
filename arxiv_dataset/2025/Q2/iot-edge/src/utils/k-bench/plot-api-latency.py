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


distributions = ['k0s', 'k3s', 'k8s', 'kubeEdge', 'openYurt']
values = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

# Create the dictionary
mapping = dict(zip(distributions, values))

# this is the data format
# k8s,operations,tests,test-numbers,metrics,medians,mins,maxs
# k0s,namespace,cp_light_1client,1,create,17.026,13.391,17.026
# k0s,namespace,cp_light_1client,1,get,1.209,1.186,1.209
# k0s,namespace,cp_light_1client,1,list,1.525,1.353,1.525

def plot_diagram(file_path, test, saveFile=False):

    all_data = []
    data = pd.read_csv(file_path)

    df = pd.DataFrame(data)

    df['k8s'] =  df['k8s'].map(mapping)

    # Prepare DataFrame for plotting by melting it
    df_melted = df.melt(id_vars=["k8s", "operations", "metrics"], 
                        value_vars=["medians", "mins", "maxs"],
                        var_name="statistic", value_name="value")

    colors = ['b', 'r', 'g', 'm', 'y']
    g = sns.FacetGrid(df_melted, col="operations", height=7, aspect=1, col_wrap=2, sharey=False, sharex=False)
    g.map_dataframe(sns.boxplot, x="metrics", y="value", hue="k8s", showfliers=False, palette=colors)
    g.add_legend()
    g.set_titles("{col_name} operation " + f"for {test}")
    g.set_axis_labels("Operations", "Latency (ms)")
    if saveFile:
        plt.savefig(f'../../diagrams/latency-statistics/{test}.pdf', format='pdf')
    else:
        plt.show()


# tests=["cp_light_1client", "cp_heavy_8client", "cp_heavy_12client"]
tests=["cp_heavy_12client"]

for test in tests:
    plot_diagram(f"../../k-bench-results/latency-statistics/{test}.csv",test, saveFile=False)
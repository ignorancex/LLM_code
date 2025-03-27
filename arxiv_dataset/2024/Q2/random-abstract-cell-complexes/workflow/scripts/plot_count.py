import pandas as pd
import seaborn as sns
from snakemake.script import Snakemake
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import networkx as nx
from collections import defaultdict

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

x = snakemake.params.get('x', 'est')
norm = snakemake.params.get('norm', None)
y = snakemake.params.get('y', 'count')
ratio = snakemake.params.get("ratio", "false") == 'True'
count_inset = snakemake.params.get("count_inset", "True") == 'True'
relative_inset = snakemake.params.get("relative_inset", "True") == 'True'
x_label = snakemake.params.get("x_label", None)
y_label = snakemake.params.get("y_label", None)
red_line_data = snakemake.params.get("red_line", None)

df_est = pd.read_csv(snakemake.input[0])
df_est['div'] = df_est['count'] / df_est['est']
min_count = df_est[x].min() 
max_count = df_est[x].max()

sns.set_theme('talk')

def plot_on_ax(df_data, ax, x_axis, y_axis, ratio, normalize=None, x_label=None, y_label=None, title=None, drop_zeros=False, opt_line= True, auto_ylog=True):
    min_count = df_data[x_axis].max()
    max_count = df_data[x_axis].min()

    if normalize is not None:
        df_data[y_axis] /= df_data[normalize]
        #df_data[y_axis] -= 1
    
    if drop_zeros:
        df_data = df_data[df_data[y_axis] != 0]


    plot = sns.lineplot(
        data=df_data, x=x_axis, y=y_axis, units='run', legend=False,
        estimator=None, alpha=0.2,
        #height=2.6, aspect=1.4,
        ax=ax, size=10
    )

    if ratio:
        if opt_line:
            plt.plot([min_count, max_count], [1, 1], 'r-')
        if df_est[y_axis].max() / df_est[y_axis][df_est[y_axis] > 0].min() > 100 and auto_ylog or df_est[y_axis].max() > 20:
            plot.set(yscale='log') 
            if plot.get_ylim()[1] < 10:
                maxval = df_data[y_axis].max()
                minval = df_data[y_axis][df_est[y_axis] > 0].min()
                ticks = [min([x for x in [.2, .5, 1] if minval <= x]), 1, max([x for x in [1, 2, 5] if x <= maxval])]
                plot.set_yticks(ticks, ticks)
    else:
        if opt_line:
            plt.plot([min_count, max_count], [min_count, max_count], 'r-')
        plot.set(yscale='log')
    
    if x_axis != 'l':
        plot.set(xscale='log')

    if y_label is not None:
        plot.set_ylabel(y_label)
    if x_label is not None:
        plot.set_xlabel(x_label)
    if title is not None:
        plot.set_title(title)
    #plot.set_axis_labels(r'edge noise ($\sigma_n$)', r'correct')
    return plot

fig, ax = plt.subplots()

plot = plot_on_ax(df_est.copy(), ax, x, y, ratio, norm, drop_zeros= x != 'l', x_label=x_label, y_label=y_label, opt_line=red_line_data is None)
if red_line_data is not None:
    ax.plot(df_est[df_est.run == 0][x], df_est[df_est.run == 0][red_line_data], 'r-')
if relative_inset:
    axins = inset_axes(plot, "35%", "40%", 'lower right', bbox_to_anchor=(0,0.15,1,1), bbox_transform=plot.transAxes)
    inset_plot = plot_on_ax(df_est.copy(), axins, 'l', y, True, 'count', 'len', '', 'rel. error', auto_ylog=False)
if count_inset:
    axins2 = inset_axes(plot, "35%", "32%", 'upper left', bbox_to_anchor=(0.1,0,1,0.95), bbox_transform=plot.transAxes)
    inset_plot = plot_on_ax(df_est.copy(), axins2, 'l', 'sample_occured', True, None, 'len', '', 'sampled cycles', opt_line=False, drop_zeros=True)
    #axins2.plot(df_est[df_est.run == 0].l, df_est[df_est.run == 0].a_priori, 'r-')


fig.tight_layout(pad=0.2)
fig.savefig(snakemake.output[0])
import pandas as pd
import seaborn as sns
from snakemake.script import Snakemake
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])
y_axis = snakemake.params.get("y_axis", "estimated")
y_label = snakemake.params.get("y_label", None)
x_label = snakemake.params.get("x_label", None)
ratio = snakemake.params.get("ratio", "false") == 'True'
relative_inset = snakemake.params.get("relative_inset", "false") == 'True'
relative_inset_y_log = snakemake.params.get("relative_inset_y_log", "false") == 'True'
relative_inset_pos = snakemake.params.get("relative_inset_pos", 'lower right')
y_log = snakemake.params.get("y_log", "false") == 'True'

mean_per_length = df_data.groupby('len').agg({'correct': 'mean'}).rename({'correct': 'mean_per_length'},axis=1)
df_data = df_data.join(mean_per_length, 'len')


sns.set_theme('talk')
def plot_on_ax(df_data, ax, y_axis, ratio, y_log, x_label=None, y_label=None, title=None, cbar=False):
    max_prob = df_data.correct.max() * 2
    min_prob = df_data.correct.min() / 2

    if ratio:
        df_data[y_axis] /= df_data['correct']
    
    if len(df_data.same_cluster.unique()) > 1:
        df_data.sort_values('same_cluster', inplace=True)


    plot = sns.scatterplot(
        data=df_data, x="correct", y=y_axis, hue='len', palette='viridis', legend=False,
        #height=2.6, aspect=1.4,
        style='same_cluster' if len(df_data.same_cluster.unique()) > 1 else None,
        ax=ax
    )

    normalize = mpl.colors.Normalize(df_data.len.min(), df_data.len.max())

    if cbar:
        cbar = plt.colorbar(cm.ScalarMappable(normalize, 'viridis'), ax=plot)
        cbar.set_label('cycle length')

    if ratio:
        plot.plot([min_prob, max_prob], [1, 1], 'r-')
        plot.set(xscale='log')
        if y_log:
            plot.set(yscale='log')
            if df_data[y_axis].max() < 10 and df_data[y_axis].min() > 0.1:
                ticks = [2**i for i in range(-3,4) if 2**i > df_data[y_axis].min() and 2**i < df_data[y_axis].max()]
                plot.set_yticks(ticks)
                plot.set_yticklabels(ticks)
                    
    else:
        plot.plot([min_prob, max_prob], [min_prob, max_prob], 'r-')
        plot.set(yscale='log', xscale='log')

    if y_label is not None:
        plot.set_ylabel(y_label)
    if x_label is not None:
        plot.set_xlabel(x_label)
    if title is not None:
        plot.set_title(title)
    return plot

if not relative_inset:
    fig, ax = plt.subplots()

    plot = plot_on_ax(df_data.copy(), ax, y_axis, ratio, y_log, x_label, y_label, cbar=True)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.25,3.6))

    plot = plot_on_ax(df_data.copy(), ax1, y_axis, ratio, y_log, x_label, y_label, cbar=False)
    #axins = inset_axes(plot, "35%", "40%", relative_inset_pos, bbox_to_anchor=(0.14,0.07,1,.88), bbox_transform=plot.transAxes)
    inset_plot = plot_on_ax(df_data.copy(), ax2, y_axis, True, relative_inset_y_log, '', '', 'rel. error', cbar=True)

fig.tight_layout(pad=0.2)
fig.savefig(snakemake.output[0])
import pandas as pd
import seaborn as sns
from snakemake.script import Snakemake

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])

sns.set_theme()

plot = sns.relplot(
    data=df_data, kind="line", x="n", y="runtime", marker='o', style='p',
    height=2.2, aspect=1.61803, hue='method', errorbar='sd'
)

plot.set(xscale='log')
plot.set(yscale='log')
plot.set(xlabel=r'size ($n$)')
plot.set(ylabel='time [s]')

plot.savefig(snakemake.output[0])
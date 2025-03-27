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

plot_args = snakemake.params.plot_args

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])

sns.set_theme('paper')

plot = sns.relplot(
    data=df_data, kind="line", **plot_args
    #height=2.2, aspect=1.61803,
)

plot.savefig(snakemake.output[0])
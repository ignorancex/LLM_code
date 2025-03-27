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

max_orientable = df_data[df_data.orientable].N_act.max()

df_data_deduplicated = df_data[df_data.N_act <= max_orientable + 2] \
        .groupby(by=['N_act', 'run']).first().reset_index()

sns.set_theme('paper')

plot = sns.relplot(
    df_data_deduplicated, kind='line', x='N_act', y='orientable', errorbar=None,
    #height=2.2, aspect=1.61803,
    hue='model',
)

plot.savefig(snakemake.output[0])



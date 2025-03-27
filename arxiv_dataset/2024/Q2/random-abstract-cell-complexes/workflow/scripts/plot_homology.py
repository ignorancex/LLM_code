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


df_data_hom1 = df_data.copy()
df_data_hom1['dim'] = df_data_hom1.hom1
df_data_hom1['homology'] = 1

df_data_hom2 = df_data.copy()
df_data_hom2['dim'] = df_data_hom2.hom2
df_data_hom2['homology'] = 2

df_data_hom = pd.concat([df_data_hom1, df_data_hom2])

sns.set_theme()

df_plot = df_data_hom[(df_data_hom.N_act > 18) & (df_data_hom.N_act < 80) & (df_data_hom.run < 10)]

plot = sns.relplot(
    df_plot, kind='line', x='N_act', y='dim', hue='run', style='homology', palette='tab10',
    height=5, aspect=1/1.61803,
    col='model',
)

plot.savefig(snakemake.output[0])
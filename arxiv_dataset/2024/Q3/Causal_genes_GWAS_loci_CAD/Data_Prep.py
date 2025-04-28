from numpy import *
import numpy as np
import pandas as pd
import sys
from collections import Counter
import os
from pathlib import Path

"""

You can give as input 
 
1. 1st argument, exposure data which you used to extract the GWAS summary data from 
the MR-Base package.

2. 2nd argument GWAS summary data/outcome data.

These two files should be the output of either Prune_Snps_LD.py or Prune_Snps_pos.py, 
with comma as a separator and the output would be a file with the format required for the
causal analysis saved in the same input directory with suffix prepared.csv

"""


def check_valid(user_input, check):
    out = None
    allowed_extensions = [".csv"]
    if check == "path":
        if not (Path(user_input).exists() and Path(user_input).is_file and Path(
                user_input).suffix in allowed_extensions):
            print(
                "Provide a valid path with .csv extension")
            sys.exit(1)
        else:
            out = user_input
    return out


try:
    file_EX = sys.argv[1]
    file_EY = sys.argv[2]
    file_EX = check_valid(file_EX, "path")
    file_EY = check_valid(file_EY, "path")
except IndexError:
    print("You failed to either provide the minimum or maximum number of parameters required to run this code. Please "
          " read the documentation for more details")
    sys.exit(1)

path = os.path.dirname(file_EX)

name_EX = os.path.splitext(os.path.basename(file_EX))[0]
name_EY = os.path.splitext(os.path.basename(file_EY))[0]

Data_out = pd.read_csv(file_EY, sep=',', low_memory=False)
Data_exp = pd.read_csv(file_EX, sep=',', low_memory=False)
Data_out = Data_out.sort_values(by='pval.outcome', ascending=True)
Data_out.reset_index(drop=True, inplace=True)

Data_exp = Data_exp.loc[Data_exp['SNP'].isin(np.intersect1d(Data_exp.SNP, Data_out.SNP))]
Data_out = Data_out.loc[Data_out['SNP'].isin(np.intersect1d(Data_out.SNP, Data_exp.SNP))]


Data_exp.reset_index(drop=True, inplace=True)
Data_out.reset_index(drop=True, inplace=True)

unique_genes = Counter(Data_exp['exposure']).keys()
no_genes = len(unique_genes)
most_common = Counter(Data_exp['exposure']).most_common(no_genes)
genes = []
for j in range(no_genes):
    genes.append(most_common[j][0])
column_names = genes
row_names = []
snp_values = list(Data_out.SNP)
eff_values = Data_out['effect_allele.outcome'].tolist()
other_values = Data_out['other_allele.outcome'].tolist()
# Loop through the lists and append elements together with underscores
for i in range(len(snp_values)):
    combined_element = f'{snp_values[i]}_{eff_values[i]}_{other_values[i]}'  # Using an f-string to concatenate the elements
    row_names.append(combined_element)

matrix = np.zeros((len(row_names), len(column_names)))

R_data = pd.DataFrame(matrix, columns=column_names, index=row_names, dtype='object')
s_data = pd.DataFrame(np.zeros(len(row_names)), columns=['outcome'], index=row_names, dtype='object')

selected_columns = Data_out['effect_allele.outcome']
new_df = selected_columns.copy()

for n in range(no_genes):
    name = genes[n]
    exp = (most_common[n][0])
    Data = Data_exp[Data_exp['exposure'] == exp]
    Data.reset_index(drop=True, inplace=True)
    o = Data['SNP'].values
    for j1 in range(int(most_common[n][1])):
        value = o[j1]
        idx = Data_out[Data_out['SNP'] == value].index.values
        string_allele = new_df[idx[0]]
        string_allele = string_allele.replace("'", "")
        a = Data.at[j1, 'effect_allele.exposure']
        b = Data.at[j1, 'other_allele.exposure']
        if string_allele != Data.at[j1, 'effect_allele.exposure']:
            Data.at[j1, 'beta.exposure'] = - Data.at[j1, 'beta.exposure']
            Data.at[j1, 'effect_allele.exposure'] = b
            Data.at[j1, 'other_allele.exposure'] = a
        value_alleles = o[j1] + "_" + Data_out.at[idx[0], 'effect_allele.outcome'] + "_" + Data_out.at[
            idx[0], 'other_allele.outcome']
        R_data.at[value_alleles, name] = Data.at[j1, 'beta.exposure']
        s_data.at[value_alleles, 'outcome'] = Data_out.at[idx[0], 'beta.outcome']

R_data = R_data.replace(np.nan, 0)
s_data = s_data.replace(np.nan, 0)
df_R = pd.DataFrame(data=R_data)
df_s = pd.DataFrame(data=s_data)
RandS = pd.concat([df_R, df_s], axis=1)
RandS.index.name = 'SNPs'

RandS.to_csv(path + "/" + name_EX + "_prepared.csv", sep=",")
print("Prepared Datasets have been saved in the directory " + path)


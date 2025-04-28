from numpy import *
import numpy as np
import pandas as pd
import sys
import os
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from pathlib import Path
from collections import Counter

ro.r['options'](warn=-1)
base = importr('base')
base.warnings()

"""This function takes as input the 1.harmonized data file which is in format from TwoSampleMR Package.

To run this file, type in the terminal

python3 Prune_Snps_LD.py "/home/user/data.csv" 1 109817192 0.01 5E-8

Here,

Compulsory argument

1. 1st argument "/home/user/data.csv" , path to the harmonized data file

2. 2nd argument chromosome (chromosome of the SNP around which you want to get the data for analysis)

Additionally (optional),

3. 3rd argument position. (position of the SNP around which you want to get the data for analysis)

4. 4th argument Lower LD threshold (float, example 0.01), 
   Snps with LD less than or equal to 0.01 with lead Snp will not be included. default: 0.0

5. 5th argument pvalue threshold, where we will remove SNPs whose pvalue is higher in the GWAS outcome data
   than the mentioned pvalue threshold.

If you do not give the  3rd argument, the lead SNP (SNP with the lowest p-value in the GWAS data is chosen)

If you do not give the 4th and 5th arguments then default values are:

LD_threshold_lower = 0.0
pvalue = 5E-8



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
    elif check == "LD":
        if not (0.5 >= user_input >= 0.0):
            print(
                "Provide threshold in the range, 0.0 >= LD_threshold_lower <= 0.5. Please "
                " read the documentation for more details.")
            sys.exit(1)
        else:
            out = user_input

    elif check == "pvalue":
        out = user_input

    elif check == "chromosome":
        if not user_input <= 1 and user_input >= 21:
            print(
                "Please provide a valid chromosome.")
        else:
            out = user_input
    elif check == "position":
        out = user_input
    else:
        print("No parameter given")
    return out


try:
    file_EX1 = sys.argv[1]
    file_EX1 = check_valid(file_EX1, "path")
    chromosome1 = int(sys.argv[2])
    chromosome1 = check_valid(chromosome1, "chromosome")
    try:
        position1 = str(int(sys.argv[3]))
        position1 = check_valid(position1, "position")
    except ValueError:
        print(
            "Provide an integer value for position. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        position1 = None

    try:
        LD_threshold_lower1 = float(sys.argv[4])
        LD_threshold_lower1 = check_valid(LD_threshold_lower1, "LD")
    except ValueError:
        print(
            "Provide a valid value for the lower LD threshold. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        LD_threshold_lower1 = 0.0
    try:
        pvalue1 = float(sys.argv[5])
        pvalue1 = check_valid(pvalue1, "pvalue")
    except ValueError:
        print(
            "Provide a valid value for pvalue threshold. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        pvalue1 = 5E-8

except ValueError:
    print(
        "Provide a valid value for chromosome. Please "
        " read the documentation for more details.")
    sys.exit(1)

except IndexError:
    print("You failed to either provide the minimum or maximum number of parameters required to run this code. Please "
          " read the documentation for more details")
    sys.exit(1)

path_original = os.path.dirname(file_EX1)
name_EX = os.path.splitext(os.path.basename(file_EX1))[0]
LD_threshold_upper = 1.0


def SNPs_LDrange(file_EX, chromosome, position, LD_threshold_lower, pvalue):

    # read the harmonized dataset into a pandas dataframe
    Data = pd.read_csv(file_EX, sep=',', low_memory=False)
    if 'chr.x' in Data.columns and 'pos.x' in Data.columns:
        Data = Data.rename(columns={'chr.x': 'chr', 'pos.x': 'pos'})
    # choose only non-palindromic data
    subset = Data[~Data['palindromic']]
    subset.reset_index(drop=True, inplace=True)

    # Extract outcome columns
    outcome_columns = [col for col in subset.columns if '.outcome' in col]

    # Extract exposure columns
    exposure_columns = [col for col in subset.columns if '.exposure' in col]

    # Create Data_outcome DataFrame
    Data_outcome = subset[['SNP', 'chr', 'pos'] + outcome_columns]

    # Create Data_outcome DataFrame
    Data_outcome = Data_outcome.drop_duplicates(subset='SNP', keep='first').reset_index(drop=True)
    Data_outcome.reset_index(drop=True, inplace=True)
    # Create Data_exposure DataFrame
    Data_exp = subset[['SNP', 'chr', 'pos'] + exposure_columns + ['exposure']]
    Data_exp.reset_index(drop=True, inplace=True)
    Data_outcome = Data_outcome[
        ["SNP", "effect_allele.outcome", "other_allele.outcome", "beta.outcome", "pval.outcome", "chr", "pos"]]
    Data_out = Data_outcome[
        Data_outcome['chr'].astype(str) == str(chromosome)]  # Convert 'chr' to string for comparison
    Data_out.reset_index(drop=True, inplace=True)
    # choose significant SNPs
    Data_out = Data_out[Data_out['pval.outcome'] <= pvalue]
    Data_out.reset_index(drop=True, inplace=True)

    # sort by significant SNPs
    Data_out = Data_out.sort_values(by='pval.outcome', ascending=True)
    Data_out.reset_index(drop=True, inplace=True)
    if position is None:
        Data_out_new = Data_out
    else:

        # Find the index of the value in the 'pos' column
        index = Data_out[Data_out['pos'].astype(str) == str(position)].index
        if index.empty:
            error_message = (f"Value {position} not found in the 'pos' column. "
                             "Please provide a valid position.")
            sys.exit(error_message)
        # Extract the row with the found index
        row_to_move = Data_out.loc[index[0]]

        # Drop the row from the DataFrame
        Data_out = Data_out.drop(index)

        # Reorder the DataFrame with the extracted row as the first row
        df = pd.concat([row_to_move.to_frame().T, Data_out])

        # Reset the index of the DataFrame
        Data_out_new = df.reset_index(drop=True)

    if not Data_out_new.empty:
        snps = list(Data_out_new.loc[:, 'SNP'].values)
        if len(snps) != 1:
            cov_data = r("TwoSampleMR::ld_matrix")(snps, with_alleles=False, pop="EUR")

            pandas2ri.activate()
            pd_from_r_df = ro.conversion.rpy2py(cov_data)
            cov_data = pd.DataFrame(pd_from_r_df, columns=r('colnames')(cov_data),
                                    index=r('colnames')(cov_data), dtype='float')

            cov_data = pd.DataFrame.abs(cov_data)
            upper_tri = cov_data.where(np.triu(np.ones(cov_data.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if
                       any(upper_tri[column] >= float(LD_threshold_upper))]
            cov_data = cov_data.drop(labels=to_drop, axis=1)
            cov_data = cov_data.drop(labels=to_drop, axis=0)

            upper_tri = cov_data.where(np.triu(np.ones(cov_data.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if
                       upper_tri[column][0] <= float(LD_threshold_lower)]
            cov_data = cov_data.drop(labels=to_drop, axis=1)
            cov_data = cov_data.drop(labels=to_drop, axis=0)

            Data_out_new = Data_out_new[Data_out_new['SNP'].isin(cov_data.index.values)]

            Data_exp1 = Data_exp.loc[
                Data_exp['SNP'].isin(np.intersect1d(Data_exp.SNP, Data_out_new.SNP))]
            Data_out1 = Data_out_new.loc[
                Data_out_new['SNP'].isin(np.intersect1d(Data_out_new.SNP, Data_exp.SNP))]
            Data_exp1.reset_index(drop=True, inplace=True)
            Data_out1.reset_index(drop=True, inplace=True)
            pandas2ri.deactivate()
        else:
            Data_exp1 = Data_exp.loc[
                Data_exp['SNP'].isin(np.intersect1d(Data_exp.SNP, Data_out_new.SNP))]
            Data_out1 = Data_out_new.loc[
                Data_out_new['SNP'].isin(np.intersect1d(Data_out_new.SNP, Data_exp.SNP))]
            Data_exp1.reset_index(drop=True, inplace=True)
            Data_out1.reset_index(drop=True, inplace=True)
        if not Data_out1.empty:
            path_new = os.path.join(path_original, "Datasets")
            path_RandS = path_new
            Data_preparation(Data_exp1, Data_out1, path_RandS, chromosome, position)
        else:
            sys.exit('Error Message : Dataset is empty after pruning for LD.')
    else:
        sys.exit('Error Message : Dataset is empty. There were no SNPs that passed the p-value threshold.')


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f" ")


def Data_preparation(Data_exp, Data_out, pathRS, chromosome, position):
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

    # Harmonisation of the alleles in exposure and outcome dataset, this is done already in the  TwoSampleMR Package
    # but here we check if the harmonisation is correct

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
    create_directory_if_not_exists(pathRS)
    RandS.to_csv(pathRS + "/" + name_EX + "_" + str(chromosome) + "_" + str(int(position)) + ".csv",
                 sep=",")
    print("Prepared Datasets have been saved in the directory " + pathRS)


SNPs_LDrange(file_EX1, chromosome1, position1, LD_threshold_lower1, pvalue1)


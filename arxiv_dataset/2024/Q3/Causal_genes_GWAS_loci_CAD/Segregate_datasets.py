from numpy import *
import numpy as np
import pandas as pd
import sys
from collections import Counter
import os
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from pathlib import Path

ro.r['options'](warn=-1)
base = importr('base')
base.warnings()

"""This function takes files which have harmonized exposure and outcome data across Chromosomes and divides the dataset 
into sub-datasets with each sub-dataset having locus-specific data

To run this file, type in the terminal

python3 Segregate_datasets.py "/home/user/Exposure.csv" 500000 0.01 1.0

Here,

Compulsory argument

1. 1st argument "/home/user/data.csv" , path to the harmonized data file

Additionally (optional),

2. 2nd argument distance_threshold, (int, example 250000 for 0.25 Mb radius), 
   Snps within this distance around the lead Snp will be chosen for a given dataset. default: 500000
   
3. 3rd argument Lower LD threshold (float, example 0.01), Snps with LD less than or equal to 0.01 
   with lead Snp will not be included. default: 0.00

4. 4th argument pvalue threshold, where we will remove SNPs whose pvalue is higher in the GWAS outcome data
   than the mentioned pvalue threshold.

If you do not give the 2nd, 3rd and 4th arguments then default values are:

position =  500000
LD_threshold_lower = 0.01
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

    elif check == "distance":
        if not (1000000 > user_input > 25000):
            print(
                "Provide distance in the range, 25000 > distance < 1000000. Please "
                " read the documentation for more details.")
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
    else:
        print("No parameter given")
    return out


try:
    file_EX1 = sys.argv[1]
    file_EX1 = check_valid(file_EX1, "path")
    try:
        distance1 = int(sys.argv[2])
        distance1 = check_valid(distance1, "distance")
    except ValueError:
        print(
            "Provide a non-zero integer value for distance. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        distance1 = 500000
    try:
        LD_threshold_lower1 = float(sys.argv[3])
        LD_threshold_lower1 = check_valid(LD_threshold_lower1, "LD")
    except ValueError:
        print(
            "Provide a valid value for the lower LD threshold. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        LD_threshold_lower1 = 0.00
    try:
        pvalue1 = float(sys.argv[4])
        pvalue1 = check_valid(pvalue1, "pvalue")
    except ValueError:
        print(
            "Provide a valid value for pvalue threshold. Please "
            " read the documentation for more details.")
        sys.exit(1)
    except IndexError:
        pvalue1 = 5E-8
except IndexError:
    print("You failed to either provide the minimum or maximum number of parameters required to run this code. Please "
          " read the documentation for more details")
    sys.exit(1)

path_original = os.path.dirname(file_EX1)
name_EX = os.path.splitext(os.path.basename(file_EX1))[0]
LD_threshold_upper = 1.0


def get_datasets(file_EX, distance, LD_threshold_lower, pvalue):
    prepare = True
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
    Data_exposure = subset[['SNP', 'chr', 'pos'] + exposure_columns + ['exposure']]
    Data_exposure.reset_index(drop=True, inplace=True)
    Data_outcome = Data_outcome[
        ["SNP", "effect_allele.outcome", "other_allele.outcome", "beta.outcome", "pval.outcome", "chr", "pos"]]
    Data_outcome = Data_outcome[Data_outcome['pval.outcome'] <= pvalue]
    Data_outcome.reset_index(drop=True, inplace=True)

    if not Data_outcome.empty:

        Data_exposure = Data_exposure.loc[
            Data_exposure['SNP'].isin(np.intersect1d(Data_exposure.SNP, Data_outcome.SNP))]
        Data_outcome = Data_outcome.loc[Data_outcome['SNP'].isin(np.intersect1d(Data_outcome.SNP, Data_exposure.SNP))]
        Data_exposure.reset_index(drop=True, inplace=True)
        Data_outcome.reset_index(drop=True, inplace=True)

        if not Data_outcome.empty:

            unique_chr = Counter(Data_outcome['chr']).keys()
            no_chr = len(unique_chr)
            get_chr = Counter(Data_outcome['chr']).most_common(no_chr)

            for i in range(no_chr):
                Dataframe = (Data_outcome.loc[Data_outcome['chr'] == get_chr[i][0]])
                Dataframe.reset_index(drop=True, inplace=True)
                Dataframe = Dataframe.sort_values(by='pval.outcome', ascending=True)
                Dataframe.reset_index(drop=True, inplace=True)
                chromosome = get_chr[i][0]
                if not Data_outcome.empty:
                    while not Dataframe.empty:

                        # choose SNPs within 'distance' around the lead SNP ######################
                        position = Dataframe.at[0, 'pos']
                        lower = int(position) - int(distance)
                        upper = int(position) + int(distance)
                        Data_out_pos = Dataframe[(Dataframe['pos'] >= lower) & (Dataframe['pos'] <= upper)]
                        Data_out_pos.reset_index(drop=True, inplace=True)
                        Dataframe = Dataframe.loc[~Dataframe['SNP'].isin(Data_out_pos['SNP'])].dropna()
                        Dataframe.reset_index(drop=True, inplace=True)

                        Data_out_pos = Data_out_pos.loc[
                            Data_out_pos['SNP'].isin(np.intersect1d(Data_out_pos.SNP, Data_exposure.SNP))]
                        Data_out_pos.reset_index(drop=True, inplace=True)
                        Data_exp_pos = Data_exposure.loc[
                            Data_exposure['SNP'].isin(np.intersect1d(Data_exposure.SNP, Data_out_pos.SNP))]
                        Data_exp_pos.reset_index(drop=True, inplace=True)
                        ##########################################################################

                        # prune SNPs according to LD ############################################
                        Data_exp_ld, Data_out_ld = SNPs_LDrange(Data_exp_pos, Data_out_pos, LD_threshold_lower)
                        ##########################################################################

                        if isinstance(Data_out_ld, pd.DataFrame):
                            if not Data_out_ld.empty:
                                path_new = os.path.join(path_original, "Datasets")
                                path_RandS = path_new
                                if prepare:
                                    Data_preparation(Data_exp_ld, Data_out_ld, path_RandS, chromosome, position)
                                else:
                                    continue
                            else:

                                continue
                        else:
                            continue
                else:
                    continue

            else:
                sys.exit('Error Message : Dataset is empty. There were no SNPs that passed the p-value threshold.')

        else:
            sys.exit('Error Message : Dataset is empty, no SNPs shared with the exposure data.')

    else:
        sys.exit('Error Message : Dataset is empty. There were no SNPs that passed the p-value threshold.')


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f" ")


def SNPs_LDrange(Data_exp, Data_out_pos, LD_threshold_lower):
    snps = list(Data_out_pos.loc[:, 'SNP'].values)
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
        Data_out_pos = Data_out_pos[Data_out_pos['SNP'].isin(cov_data.index.values)]

        Data_exp1 = Data_exp.loc[
            Data_exp['SNP'].isin(np.intersect1d(Data_exp.SNP, Data_out_pos.SNP))]
        Data_out1 = Data_out_pos.loc[
            Data_out_pos['SNP'].isin(np.intersect1d(Data_out_pos.SNP, Data_exp.SNP))]
        Data_exp1.reset_index(drop=True, inplace=True)
        Data_out1.reset_index(drop=True, inplace=True)
        pandas2ri.deactivate()
    else:
        Data_exp1 = Data_exp.loc[
            Data_exp['SNP'].isin(np.intersect1d(Data_exp.SNP, Data_out_pos.SNP))]
        Data_out1 = Data_out_pos.loc[
            Data_out_pos['SNP'].isin(np.intersect1d(Data_out_pos.SNP, Data_exp.SNP))]
        Data_exp1.reset_index(drop=True, inplace=True)
        Data_out1.reset_index(drop=True, inplace=True)
    return Data_exp1, Data_out1


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
    # name_EX_new = name_EX.replace("exp", "")

    RandS.to_csv(pathRS + "/" + name_EX + "_" + str(chromosome) + "_" + str(position) + ".csv",
                 sep=",")
    print("Prepared Datasets have been saved in the directory " + pathRS)


get_datasets(file_EX1, distance1, LD_threshold_lower1, pvalue1)

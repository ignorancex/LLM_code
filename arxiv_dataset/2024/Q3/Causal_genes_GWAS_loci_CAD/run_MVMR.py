from scipy.linalg import lstsq
from numpy import *
import numpy as np
import pandas as pd
import sys
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri



"""

getting the causal genes for cases the user cannot supply the LD-matrix.


To run the file, type in the command line python3 run_MVMR.py "/home/user/exposure_outcome.csv"


where: 

"/home/user/exposure_outcome.csv"

sys.argv[1] is the first argument when you run the file and should be the exposure_outcome.csv file 
containing the SNPs to exposure effects and SNPs to outcome effects.


format:

SNPs,ENSG00000143028.8,ENSG00000134222.16,outcome
rs602633_G_T,-0.247463,0.0,0.096018
rs4970834_T_C,0.263743,-0.208615,-0.09655



The output after running this file would be a .csv with results from the methods (depending
 on the dimensions, least-squares, generalized method of moments and ratio method)
saved as .csv file in the same directory as the one given for the exposure_outcome.csv file. 

i.e. for this example "/home/user/exposure_outcome_results.csv"

"""

file_EXEY = sys.argv[1]
Data = pd.read_csv(file_EXEY, sep=',')
for (columnName, columnData) in Data.iteritems():
    if (Data[columnName] == 0).all():
        Data.drop(columnName, axis=1, inplace=True)
snps_full = Data.loc[:, 'SNPs'].values
snps = [i.split('_', 1)[0] for i in snps_full]
effect_alleles = [i.split('_', 1)[1] for i in snps_full]
if len(snps) != 1:
    cov_data = r("TwoSampleMR::ld_matrix")(snps)
    pandas2ri.activate()
    pd_from_r_df = ro.conversion.rpy2py(cov_data)
    cov_data = pd.DataFrame(pd_from_r_df, columns=r('colnames')(cov_data), index=r('colnames')(cov_data), dtype='float')
    eff_all_covmatrix = [i.split('_', 1)[1] for i in r('colnames')(cov_data)]
    if (np.array(eff_all_covmatrix) != np.array(effect_alleles)).all() or (
            np.array(eff_all_covmatrix) == np.array(effect_alleles)).all():
        cov_data = cov_data
    else:
        res = (np.array(eff_all_covmatrix) != np.array(effect_alleles))
        for t in range(len(res)):
            if res[t]:
                cov_data[cov_data.columns[t]] = - cov_data[cov_data.columns[t]]
                cov_data.iloc[[t]] = -cov_data.iloc[[t]]

    upper_tri = cov_data.where(np.triu(np.ones(cov_data.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] == 1.0)]
    cov_data = cov_data.drop(labels=to_drop, axis=1)
    cov_data = cov_data.drop(labels=to_drop, axis=0)
    cov_EE = cov_data.values
    snps_new = [i.split('_', 1)[0] for i in r('colnames')(cov_data)]
    cov_data_pruned = pd.DataFrame(cov_EE, columns=snps_new, index=snps_new, dtype='float')
    snps_Data = [i.split('_', 1)[0] for i in Data.SNPs]
    Data.SNPs = snps_Data
    Data = Data[Data['SNPs'].isin(cov_data_pruned.index.values)]
    Data.reset_index(drop=True, inplace=True)
    all_zeros = (Data == 0).all()
    if all_zeros.any():
        sys.exit('Error Message : Some columns have all zero rows in the new pruned dataset ad the matrix is singular.')

    else:
        snps_final = Data.loc[:, 'SNPs'].values
        no_snps = len(snps_final)
        no_genes = len(Data.columns) - 2
        df = Data.loc[:, ~Data.columns.isin(['SNPs', 'outcome'])]
        gene_names = column_names = list(df.columns.values)
        covEY = Data.loc[:, 'outcome'].values
        covEX = df.values
        if no_snps >= no_genes:
            err = np.linalg.inv(covEX.T @ np.linalg.inv(cov_EE) @ covEX)
            standard_err1 = np.sqrt(np.diagonal(err))
        
            standard_err1 = standard_err1.tolist()
            standard_err = []
            N_gwas = 141217
        
            if no_snps == no_genes:
                if no_snps == 1:
                    b_est = covEY / covEX
                    d1 = {'gene': gene_names,
                          'Causal Estimate Least Squares': b_est[0], 'Causal Estimate GMM': b_est[0],
                          'No_snps': no_snps, 'se': standard_err}
                    df1 = pd.DataFrame(data=d1)
                    df1 = df1.set_index('gene')
                else:
                    b_est = np.linalg.solve(covEX, covEY)
                    p = len(gene_names)
                    no_snps1 = [no_snps]
                    for i in range(p - 1):
                        no_snps1.append("")
                    d1 = {'gene': gene_names,
                          'Causal Estimate Least Squares': b_est, 'Causal Estimate GMM': b_est,
                          'No_snps': no_snps1, 'se': standard_err}
                    df1 = pd.DataFrame(data=d1)
                    df1 = df1.set_index('gene')
            else:
                if no_snps > no_genes:
                    b_est2, res, rnk, sy = lstsq(covEX, covEY)
                    b_est1 = np.linalg.inv(covEX.T @ np.linalg.inv(cov_EE) @ covEX) @ (covEX.T @ np.linalg.inv(cov_EE) @ covEY)
                    no_snps1 = [no_snps]
                    p = len(gene_names)
                    for i in range(p - 1):
                        no_snps1.append("")
                    d1 = {'gene': gene_names,
                          'Causal Estimate Least Squares': b_est2, 'Causal Estimate GMM': b_est1,
                          'No_snps': no_snps1, 'se': standard_err, }
                    df1 = pd.DataFrame(data=d1)
                    df1 = df1.set_index('gene')
                else:
                    sys.exit('Error Message : You require at least as many instruments as exposures to run this '
                             'analysis.')
        else:
            sys.exit('Error Message : You require at least as many instruments as exposures to run this analysis.')


else:
    no_genes = len(Data.columns) - 2
    if len(snps) >= no_genes:
        df = Data.loc[:, ~Data.columns.isin(['SNPs', 'outcome'])]
        gene_names = column_names = list(df.columns.values)
        covEY = Data.loc[:, 'outcome'].values
        covEX = df.values
        b_est = covEY / covEX
        standard_err = np.linalg.inv(covEX.T @ covEX)
        N_gwas = 141217
        d1 = {'gene': gene_names,
              'Causal Estimate Least Squares': b_est[0], 'Causal Estimate GMM': b_est[0],
              'No_snps': 1, 'se': np.sqrt(standard_err[0])/np.sqrt(N_gwas)}
        df1 = pd.DataFrame(data=d1)
        df1 = df1.set_index('gene')
    else:
        sys.exit('Error Message : You require at least as many instruments as exposures to run this analysis.')

df1.to_csv(file_EXEY + "_results.csv", sep=",", float_format='%g')

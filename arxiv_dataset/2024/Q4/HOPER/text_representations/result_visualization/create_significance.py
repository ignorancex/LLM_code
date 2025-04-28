import pandas as pd
import glob
import os
import numpy as np
import ast
import scipy
import math
from scipy.stats import ttest_rel
import statsmodels.stats.multitest as mt

from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro

result_path = ""

#calculate_q_vals(go_pred_score_table): Calculates q-values using the Benjamini/Hochberg method.
def calculate_q_vals(go_pred_score_table):
    
    tTest_pvalue_list = []
    for model1 in go_pred_score_table:
        for model2 in go_pred_score_table:
            t_result =scipy.stats.wilcoxon(go_pred_score_table[model1], go_pred_score_table[model2],zero_method='zsplit')
            tTest_pvalue_list.append(t_result.pvalue)

    tTest_pvalue_list = [nan_to_zero(p_val) for p_val in tTest_pvalue_list]
    #multiple test correction using  Benjamini/Hochberg
    q_vals = mt.multipletests(tTest_pvalue_list, method='fdr_bh')[1]

    #convert q value list to dataframe
    model_number = len(go_pred_score_table.columns)
    chunks = [q_vals[x:x+model_number] for x in range(0, len(q_vals), model_number)]
    qval_df = pd.DataFrame(chunks,columns=go_pred_score_table.columns)

    qval_df.set_index(go_pred_score_table.columns, inplace=True)
    return qval_df

#check_for_normality(go_pred_signinificance_score_df): Checks if data is drawn from a normal distribution.
def check_for_normality(go_pred_signinificance_score_df):
    for col in go_pred_signinificance_score_df:
        stat, p = shapiro(go_pred_signinificance_score_df[col])
        # interpret
        alpha = 0.05
        if p < alpha:
            print('Data does not drawn from a Normal distribution (reject H0) for ' + col)
            print('Statistics=%.3f, p=%.3f' % (stat, p))

#nan_to_zero(x): Replaces NaN values with zero.
def nan_to_zero(x):
    if math.isnan(x):
        return 0
    else:
        return x

#create_significance_tables(): Creates significance tables for different aspects.
def create_significance_tables():
    path=os.getcwd()
    go_pred_signinificance_score_mf = pd.DataFrame()
    go_pred_signinificance_score_bp = pd.DataFrame()
    go_pred_signinificance_score_cc = pd.DataFrame()

    measure = 'F1_Weighted'
    for filename in sorted(glob.glob(os.path.join(result_path, '*_5cv_[!m|s]*'))):
        tmp_column = pd.read_csv(filename,sep="\t")
        tmp_column.sort_values(tmp_column.columns[0],inplace=True)
        col_name = filename.split("Ontology_based_function_prediction_5cv_")[-1].split(".")[0]
        score_list_mf = []
        score_list_bp = []
        score_list_cc = []
        #print(col_name)
        for index,row in tmp_column.iterrows():
            score = row[measure]
            #print(score)
            if 'MF' in row.iloc[0]:
                score_list_mf += ast.literal_eval(score)
            if 'BP' in row.iloc[0]:
                score_list_bp += ast.literal_eval(score)
            if 'CC' in row.iloc[0]:
                score_list_cc += ast.literal_eval(score)
        go_pred_signinificance_score_mf[col_name] = score_list_mf
        go_pred_signinificance_score_bp[col_name] = score_list_bp
        go_pred_signinificance_score_cc[col_name] = score_list_cc
	


    check_for_normality(go_pred_signinificance_score_mf)
    check_for_normality(go_pred_signinificance_score_bp)
    check_for_normality(go_pred_signinificance_score_cc)


    mf_qval_df = calculate_q_vals(go_pred_signinificance_score_mf)
    bp_qval_df = calculate_q_vals(go_pred_signinificance_score_bp)
    cc_qval_df = calculate_q_vals(go_pred_signinificance_score_cc)


    mf_qval_df.to_csv(os.path.join(path,"text_representations/result_visualization/significance/mf_qval_df.csv"))
    bp_qval_df.to_csv(os.path.join(path,"text_representations/result_visualization/significance/bp_qval_df.csv"))
    cc_qval_df.to_csv(os.path.join(path,"text_representations/result_visualization/significance/cc_qval_df.csv"))

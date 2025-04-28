import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from flask import Flask, render_template, request


def ImputRNA(df):
    # create an instance of the KNN imputer
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

    # perform the imputation on your dataframe
    df_imputed = imputer.fit_transform(df)

    # fit_transform returns a numpy array, so if you want it back in a dataframe you can do:
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    
    return(df_imputed)


def data_sampling(clinical_filter_patient, ratio):
    
    # Function to filter training samples and preprocess data
    clinical_filter_patient_1 = clinical_filter_patient[clinical_filter_patient["OS_STATUS"] == 1]["PATIENT_ID"]
    clinical_filter_patient_0 = clinical_filter_patient[clinical_filter_patient["OS_STATUS"] == 0]["PATIENT_ID"]
    
    train_patients_1, test_patients_1 = train_test_split(clinical_filter_patient_1, test_size=(1 - ratio))
    train_patients_0, test_patients_0 = train_test_split(clinical_filter_patient_0, test_size=(1 - ratio))
    
    train_patients = pd.concat([train_patients_0, train_patients_1]).tolist()
    test_patients = pd.concat([test_patients_0, test_patients_1]).tolist()
    
    classified_patients = {"train_patients": train_patients, "test_patients": test_patients}
    
    return classified_patients


def sampled_data_preprocess(chosen_patients, clinical_filter_patient, rna_filter_nor_info, chosen_genes):
    clinical_filter_patient.index = clinical_filter_patient["PATIENT_ID"]
    clinical_info = clinical_filter_patient.loc[chosen_patients, ["OS_STATUS", "OS_MONTHS"]]
    rna_info = rna_filter_nor_info.loc[chosen_genes, chosen_patients].T
    
    survival_info = pd.concat([clinical_info, rna_info], axis=1)
    
    return survival_info


def uni_select_genes(survival_info):   
    # Function to select genes using univariate Cox regression
    
    covariates = survival_info.columns[2:]
    univ_results = []
    
    cph = CoxPHFitter()
    for covariate in covariates:
        df = survival_info.loc[:, ['OS_MONTHS', 'OS_STATUS', covariate]]
        df['OS_STATUS'] = df['OS_STATUS'].astype(int)
        
        cph.fit(df, duration_col='OS_MONTHS', event_col='OS_STATUS')
        summary = cph.summary
        summary["covariates"] = covariate
        univ_results.append(summary)
    
    # Concatenate all summaries into a single DataFrame
    res = pd.concat(univ_results)
    res = res.drop(columns = ['covariates'])
    uni_cox = res[res['p'] < 0.05]
    uni_selected_genes = res[res['p'] < 0.05].index.tolist()
    
    uniCox_result = {"uni_res": res, "uni_genes": uni_selected_genes}   
    
    return uniCox_result


def construct_model(candidate_genes, survival_info):
    # Convert candidate_genes to a string formula
    genes_formula = ' + '.join(candidate_genes)
    full_formula = 'OS_MONTHS ~ ' + genes_formula
    
    survival_info = survival_info.dropna()   
    
    # Filter survival_info for required columns
    survival_info_filtered = survival_info[['OS_MONTHS', 'OS_STATUS'] + candidate_genes].copy()
    survival_info_filtered['OS_STATUS'] = survival_info_filtered['OS_STATUS'].astype(int)

    # Fit Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(survival_info_filtered, 'OS_MONTHS', event_col='OS_STATUS')
    
    # Get the summary of the model fit
    multi_cox_res = cph.summary

    # Get the coefficients for the candidate genes
    genes_coef = multi_cox_res.loc[candidate_genes, 'exp(coef)']

    # Create the formula string
    formula_str = [f'{coef:.3f}*{gene}' for coef, gene in zip(genes_coef, candidate_genes)]
    formula_exp = ' + '.join(formula_str)
    
    model_res = {"cox_res": multi_cox_res,
                 "coef": genes_coef,
                 "formula": formula_exp}
    
    return model_res


def validate_model(candidate_genes, survival_info, coef, plot_prefix):
    # Calculate ORS
    survival_info['ORS'] = survival_info[candidate_genes].values @ coef.values
    # This command is equal to the following:
    # survival_info['ORS'] = survival_info[candidate_genes].dot(coef)

    # Define cutpoint for ORS
    ORS_threshold = survival_info['ORS'].median()
    survival_info['ORS_group'] = np.where(survival_info['ORS'] > ORS_threshold, "High", "Low")
    
    # Perform survival analysis
    kmf = KaplanMeierFitter()
    kmf.fit(survival_info['OS_MONTHS'][survival_info['ORS_group'] == 'High'], event_observed=survival_info['OS_STATUS'][survival_info['ORS_group'] == 'High'], label='High')
    ax = kmf.plot(ci_show=True, color = "red")

    kmf.fit(survival_info['OS_MONTHS'][survival_info['ORS_group'] == 'Low'], event_observed=survival_info['OS_STATUS'][survival_info['ORS_group'] == 'Low'], label='Low')
    kmf.plot(ax=ax, ci_show=True, color = "darkblue")

    # Perform log-rank test
    results = logrank_test(survival_info['OS_MONTHS'][survival_info['ORS_group'] == 'High'], survival_info['OS_MONTHS'][survival_info['ORS_group'] == 'Low'], 
survival_info['OS_STATUS'][survival_info['ORS_group'] == 'High'], survival_info['OS_STATUS'][survival_info['ORS_group'] == 'Low'], alpha=.95)
    # results.print_summary()

    p_value = results.p_value

    # Plotting
    plt.xlabel('Time')
    plt.ylabel('Survival probability')
    plt.legend(title='Risk', loc='upper right')
    plt.text(0.15, 0.1, f'P value: {p_value:.3f}', 
     horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    plt.savefig(f"{plot_prefix}.png")
    plt.savefig(f"{plot_prefix}.pdf", format='pdf')    

    plt.clf()

    ORS_threshold = round(ORS_threshold, 3)
    p_value = round(p_value, 3)

    # surv_info = {"threshold": ORS_threshold}
    surv_info = {"threshold": ORS_threshold,
				"pValue": p_value}

    return surv_info

def MulMarker(analysis_name, chosen_genes_DF, clinical_filter_patient, rna_filter_nor_info, seed, ratio):
    ############# The main code for the function ###################

	# Using KNN to impute the missing RNA expression values
    if rna_filter_nor_info.isna().any().any():
        rna_filter_nor_info = ImputRNA(rna_filter_nor_info)
            
    # Delete the patients with NA values in OS_STATUS and OS_MONTHS
    if clinical_filter_patient.isna().any().any():
         clinical_filter_patient = clinical_filter_patient.dropna()
             
    np.random.seed(seed)
    chosen_genes = chosen_genes_DF.iloc[:,0].tolist()
        
    classified_patients = data_sampling(clinical_filter_patient, ratio)
    train_patients = classified_patients['train_patients']
    test_patients = classified_patients['test_patients']
    train_survival_info = sampled_data_preprocess(train_patients, clinical_filter_patient, rna_filter_nor_info, chosen_genes)
    test_survival_info = sampled_data_preprocess(test_patients, clinical_filter_patient, rna_filter_nor_info, chosen_genes)
    total_survival_info = sampled_data_preprocess(clinical_filter_patient['PATIENT_ID'],
                                                  clinical_filter_patient, rna_filter_nor_info, chosen_genes)
    uniCox = uni_select_genes(train_survival_info)    
    uni_genes = uniCox['uni_genes']
    uni_res = uniCox['uni_res']
    multiCox = construct_model(uni_genes, train_survival_info)
    coef = multiCox['coef']
    mul_res = multiCox['cox_res']
    formula = multiCox['formula']
    train_prefix = "static/" + analysis_name + "_train_survival_plot"
    test_prefix = "static/" + analysis_name + "_test_survival_plot"
    total_prefix = "static/" + analysis_name + "_total_survival_plot"
    train_res = validate_model(uni_genes, train_survival_info, coef, train_prefix)
    test_res = validate_model(uni_genes, test_survival_info, coef, test_prefix)
    total_res = validate_model(uni_genes, total_survival_info, coef, total_prefix)

    # write the results into files
    uni_res_file_name = 'static/' + analysis_name + '_uni_res.csv'
    mul_res_file_name = 'static/' + analysis_name + '_mul_res.csv'
    uni_res.to_csv(uni_res_file_name)
    mul_res.to_csv(mul_res_file_name)
    # Move the index data to a new column
    uni_res = uni_res.reset_index()
    mul_res = mul_res.reset_index()
    uni_res_html = uni_res.to_html(index = False, classes = 'my-table')
    mul_res_html = mul_res.to_html(index = False, classes = 'my-table')

    total_res = {
                "uni_genes": uni_genes,
				"uni_res_html": uni_res_html,
				"formula": formula,
                "mul_res_html": mul_res_html,
                "uni_res_file": uni_res_file_name,
                "mul_res_file": mul_res_file_name,
                "train_prefix": train_prefix,
                "test_prefix": test_prefix,
                "total_prefix": total_prefix,
                "train_res": train_res,
                "test_res": test_res,
                "total_res":total_res
                }

    return total_res

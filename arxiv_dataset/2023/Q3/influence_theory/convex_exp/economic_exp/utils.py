"""This file supports the economics_data notebooks (cash_transfer_exp.ipynb and oregon_medicaid_exp.ipynb)
"""
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
# ------------------------------------CASH TRANSFER--------------------------------


def cash_transfer_data(data_dir):
    # Download data and use only 'Cind' < 10,000 to match original study
    data = pd.read_stata(data_dir)
    data = data[data['Cind'] < 10000]

    # Change variables to NA or other
    data['hhhage'] = data['hhhage'].replace(['97 y más'], 97)
    data['hhhage_clean1'] = np.where(
        data['hhhage'] == 'no sabe', None, data['hhhage'])
    data['hhhage_cl'] = np.where(
        data['hhhage_clean1'] == 'nr', None, data['hhhage_clean1'])
    data['hhhspouse_cl'] = np.where(
        data['hhhspouse'] == 'nr', None, data['hhhspouse'])

    # Change variables to binary
    hhhsex_cond = [(data['hhhsex'] == "hombre"),
                   (data['hhhsex'] == "mujer"), (data['hhhsex'] == 9.0)]
    hhhsex_values = [1, 0, pd.NA]
    data['hhhsex_bi'] = np.select(hhhsex_cond, hhhsex_values)

    p16_cond = [(data['p16'] == "si"), (data['p16']
                                        == "no"), (data['p16'] == "nr")]
    p16_values = [1, 0, pd.NA]
    data['p16_bi'] = np.select(p16_cond, p16_values)

    hhhalpha_cond = [(data['hhhalpha'] == "si"), (data['hhhalpha'] == "no"),
                     (data['hhhalpha'] == "nr"), (data['hhhalpha'] == 'na')]
    hhhalpha_values = [1, 0, pd.NA, pd.NA]
    data['hhhalpha_bi'] = np.select(hhhalpha_cond, hhhalpha_values)

    # Fill all NA values with mean
    data.fillna(data.mean(), inplace=True)

    # Standardize the independent variables
    x_data = data[['treatp', 'treatnp', 'hhhsex_bi',
                   'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl']]
    scaler_x = preprocessing.StandardScaler().fit(x_data)
    standardized_indep = scaler_x.transform(x_data)

    # Combine data into one dataset
    standardized_data = pd.DataFrame(standardized_indep, columns=[
                                     'treatp', 'treatnp', 'hhhsex_bi', 'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl'])
    standardized_data['Cind'] = data['Cind']
    standardized_data['t'] = data['t']
    standardized_data = standardized_data.dropna()

    x_data = pd.DataFrame(x_data, columns=[
                          'treatp', 'treatnp', 'hhhsex_bi', 'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl'])
    x_data['Cind'] = data['Cind']
    x_data['t'] = data['t']
    x_data = x_data.dropna()

    return x_data, standardized_data

# Runs n_sim number of simulations computing population and empirical IF on one contaminated point
def if_diff_n_cash(data, time, n_sim, lambda_, emp_if_lin):
    if_ls_tot = {}
    H_pop = []
    # Filter data to time period
    data = data[data['t'] == time]
    n_ls = list(np.logspace(np.log10(50), np.log10(len(data)), 6).astype(int))
    # Generate random sample of contamination data points
    influence_dfs = data.sample(n_sim, random_state=2)
    for n in n_ls:
        # Generate random training sample of size n
        sample_df = data.sample(n, random_state=1)
        infl_tot = []
        for i in range(n_sim):
            i_df = influence_dfs[i:i+1]
            x = sample_df[['treatp', 'treatnp', 'hhhsex_bi',
                           'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl']]
            y = sample_df['Cind']
            x_con = i_df[['treatp', 'treatnp', 'hhhsex_bi',
                          'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl']]
            y_con = i_df['Cind']
            if_all, H = emp_if_lin(x, y, x_con, y_con, lambda_, n)
            infl_tot.append(if_all)
            if i == 0:
                H_pop.append(H)
        if_ls_tot[n] = infl_tot

    return (if_ls_tot, n_ls, H_pop)

# standardization and cleaning results
def standardize(data):
    return ([d/data[0] for d in data])


def clean_results_cash(n_ls, results_tot, H_pop):
    bound_val = []
    mean_diff_abs_total = []
    sd_diff_abs_total = []
    for i, n in enumerate(n_ls[0:len(n_ls)-1]):
        for j in range(len(results_tot)):
            diff_abs_total = np.abs(pd.DataFrame(results_tot)[
                                    n][j] - pd.DataFrame(results_tot)[n_ls[5]][j])
            bound_val.append(
                np.dot(np.matmul(H_pop[i], diff_abs_total), np.transpose(diff_abs_total)))
        mean_diff_abs_total.append(np.mean(bound_val))
        sd_diff_abs_total.append(np.std(bound_val))

    # Clean data
    mean_diff_abs_total = standardize(mean_diff_abs_total)
    sd_diff_abs_total = standardize(sd_diff_abs_total)

    return (mean_diff_abs_total, sd_diff_abs_total)

# Get data for bound calculations
def bound_values_cash(standardized_data, time):
    # Use only data from time period 8 (same as training data used in experiment)
    t = 8
    data = standardized_data[standardized_data['t'] == time]
    x_pop = data[['treatp', 'treatnp', 'hhhsex_bi',
                  'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl']]
    y_pop = data['Cind']

    # Sample a training data point
    tr_df = data.sample(1, random_state=2)
    x_con = tr_df[['treatp', 'treatnp', 'hhhsex_bi',
                   'hectareas', 'vhhnum', 'hhhage_cl', 'hhhspouse_cl']]
    y_con = tr_df['Cind']
    return (x_pop, y_pop, x_con, y_con)

# ------------------------------------CASH TRANSFER--------------------------------

# Extract data for oregon experiment
def oregon_medicaid_data(data_descr_dir, data_survey12m_dir):
    # Download data
    data_descr = pd.read_stata(data_descr_dir)
    data_survey12m = pd.read_stata(data_survey12m_dir)

    # Dependent Variables (Yihj)
    # Change # of badday to # of goodday out of 30days
    data_survey12m['gooddays_phys_12m'] = 30 - \
        data_survey12m['baddays_phys_12m']
    data_survey12m['gooddays_ment_12m'] = 30 - \
        data_survey12m['baddays_ment_12m']
    data_survey12m['gooddays_tot_12m'] = 30 - data_survey12m['baddays_tot_12m']

    # Create bin for health_gen_12m (not poor)
    data_survey12m['health_gen_bin_fair_12m'] = np.where(
        data_survey12m['health_gen_12m'] == 'poor', 0, 1)
    depen_var = data_survey12m[["person_id", "health_gen_bin_12m", "health_gen_bin_fair_12m",
                                "health_chg_bin_12m", 'gooddays_phys_12m', 'gooddays_ment_12m', 'gooddays_tot_12m', "dep_sad_12m"]]

    # Lottery Variable (LOTTERY)
    lottery = data_descr[['person_id', 'treatment', 'household_id']]

    # Household size fixed effects, survey wave fixed effects and interaction between the two (Xih)
    fixed_var = data_survey12m[['person_id', 'hhsize_12m', 'wave_survey12m']]

    # Demographic and Economic covariates (Vih)
    dem_eco_var = data_survey12m[['person_id', 'employ_hrs_12m', 'edu_12m', "dia_dx_12m", "ast_dx_12m", "hbp_dx_12m",
                                  "emp_dx_12m", "dep_dx_12m", "ins_any_12m", "ins_ohp_12m", "ins_private_12m", "ins_other_12m", "ins_months_12m"]]

    return data_survey12m, depen_var, lottery, fixed_var, dem_eco_var

# Runs n_sim number of simulations computing population and empirical IF on one contaminated point
def if_diff_n_oregon(data, n_sim, dependent_var, emp_if_lin, emp_if_log, lambda_=.1, reg_type="linear"):
    if_ls = {}
    if_total_ls = {}
    H_pop = []
    n_ls = list(np.logspace(np.log10(50), np.log10(len(data)), 6).astype(int))
    # Get random sample of contamination data points
    influence_dfs = data.sample(n_sim, random_state=2)
    for n in n_ls:
        # Get random training sample of size n
        sample_df = data.sample(n, random_state=1)
        infl = []
        infl_total = []
        for i in range(n_sim):
            i_df = influence_dfs[i:i+1]
            x = sample_df.drop([dependent_var], axis=1)
            y = sample_df[dependent_var]
            x_con = i_df.drop([dependent_var], axis=1)
            y_con = i_df[dependent_var]
            if reg_type == "linear":
                influence, H = emp_if_lin(x, y, x_con, y_con, lambda_, n)
                infl.append(influence[0])
                infl_total.append(influence)
            else:
                influence, H = emp_if_log(x, y, x_con, y_con, lambda_)
                infl.append(influence[0])
                infl_total.append(influence)
            if i == 0:
                H_pop.append(H)
        if_ls[n] = infl
        if_total_ls[n] = infl_total
    return (if_ls, if_total_ls, n_ls, H_pop)

# Pre-process data
def preprocess_data(data):
    enc = Pipeline(steps=[
        ("encoder", preprocessing.OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=np.nan)), ])

    for c in data.columns:
        if data[c].dtypes == 'category':
            data[c] = data[c].cat.add_categories(["NA"])
            data[c] = data[c].fillna("NA")
            data[c] = enc.fit_transform(pd.DataFrame(data[c]))
    return (data)


# Extract data for each dependent variable
def dependent_var_data(dep_var):
    data_survey12m, depen_var, lottery, fixed_var, dem_eco_var = oregon_medicaid_data(
        "../data/oregon_data/oregonhie_descriptive_vars.dta", "../data/oregon_data/oregonhie_survey12m_vars.dta")
    depen_var = data_survey12m[[dep_var, "person_id"]]
    # Create covariates
    x = pd.merge(pd.merge(lottery, fixed_var, on="person_id"),
                 dem_eco_var, on="person_id")
    # Remove person ID
    x_data = x.drop(columns=["person_id"])
    # Standardize covariates
    x_data = preprocess_data(x_data)
    # Replace "person_id" to merge with dependent variables
    x_data['person_id'] = x["person_id"]
    final_data = pd.merge(depen_var, x_data, on="person_id")
    final_data = final_data.dropna()
    return (final_data.drop(columns=["person_id"]))


# Clean results from experiments
def clean_results_oregon(results, results_tot, n_ls, H_pop):
    results_mean = []
    results_sd = []
    bound_val = []
    mean_diff_abs_total = []
    sd_diff_abs_total = []
    for i, n in enumerate(n_ls[0:5]):
        results_mean.append(
            (pd.DataFrame(results)[n] - pd.DataFrame(results)[n_ls[5]]).abs().mean())
        results_sd.append(
            1.96*(pd.DataFrame(results)[n] - pd.DataFrame(results)[n_ls[5]]).abs().std())
        bound_val = []
        for j in range(len(results_tot)):
            diff_abs_total = np.abs(pd.DataFrame(results_tot)[
                                    n][j] - pd.DataFrame(results_tot)[n_ls[5]][j])
            bound_val.append(
                np.dot(np.matmul(H_pop[i], diff_abs_total), np.transpose(diff_abs_total)))
        mean_diff_abs_total.append(np.mean(bound_val))
        sd_diff_abs_total.append(np.std(bound_val))
    return (n_ls[0:5], results_mean, results_sd, mean_diff_abs_total, sd_diff_abs_total)


# Extract data for bound calculations
def bound_values_oregon(data, d):
    tr_df = data.sample(1, random_state=2)
    x_pop = data.drop([d], axis=1)
    y_pop = data[d]
    x_con = tr_df.drop([d], axis=1)
    y_con = tr_df[d]
    return (x_pop, y_pop, x_con, y_con)

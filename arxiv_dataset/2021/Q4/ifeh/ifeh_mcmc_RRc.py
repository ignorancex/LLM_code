import joblib
import pymc3 as pm
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import arviz as az
import ifeh_io as io

# ======================================================================================================================
# PREAMBLE

# The root directory:
rootdir = "."

# Input data files:
inputfile1 = "CHR_X_phot_single.dat"
inputfile2 = "LIT_hires_feh_1_X_phot.dat"
inputfile3 = "LIT_hires_feh_2_X_phot.dat"
# inputfile2 = None
# inputfile3 = None

# The name of the plot showing the predicted vs ground truth values for the initial OLS fit:
ols_residual_figure = "ifeh_mc1_poly1.4_RRc_ols_residual"
# The name of the plot showing the predicted vs ground truth values for the final Bayesian fit:
residual_plot_figure = "ifeh_mc1_poly1.4_RRc_residual"

# The names of the traceplot and pairplot produced by arviz:
traceplot_figure = "traceplot1_poly1.4_RRc"
pairplot_figure = "pairplot1_poly1.4_RRc"
output_model_file = "model_mc1_poly1.4_RRc.sav"

# Format of the output figures
figformat = 'png'

# Mark the Blazhko stars in the plots?
mark_blazhko = True

subset_expr = '(type=="RRc" or type=="RRc-BL") and keep==1 and snr>70 and phcov>0.8'

# The names of the input features:
input_feature_names = ['period', 'A1', 'A2', 'A3', 'phi21', 'phi31']

# The indices of the optimal feature set:
feature_indx = [0, 1, 2, 5]

n_poly = 1  # polynomial order of the optimal feature set

# If trim_quantiles is provided, data beyond the lower and upper quantiles (qlo, qhi)
# of the listed column names will be trimmed.
trim_quantiles = None
qlo = 0.01  # lower percentile for data rejection
qhi = 0.99  # upper percentile for data rejection

# RRc:
mean_21_phase = 9.56
mean_31_phase = 6.83


# END OF PREAMBLE
# ======================================================================================================================


def shift_phase(phase, mean_phase=5.9):
    shifted_phase = phase - np.floor((phase - mean_phase + np.pi) / 2.0 / np.pi) * 2.0 * np.pi
    # shifted_phase = phase-np.floor((phase-mean_phase)/2.0/np.pi)*2.0*np.pi

    return shifted_phase


# ======================================================================================================================
# Read in the data:


X, y, yw, df, feature_names, df_orig = \
    io.load_dataset(os.path.join(rootdir, inputfile1),
                    trim_quantiles=trim_quantiles,
                    n_poly=n_poly, plothist=False,
                    usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'A1_e', 'A2_e', 'A3_e',
                             'phi1', 'phi2', 'phi3', 'phi1_e', 'phi2_e', 'phi3_e', 'phi21', 'phi31', 'phi21_e',
                             'phi31_e',
                             'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift'],
                    input_feature_names=input_feature_names, y_col='feh', output_feature_indices=feature_indx,
                    yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'])

if inputfile2 is not None:
    X2, y2, yw2, df2, feature_names, df2_orig = \
        io.load_dataset(os.path.join(rootdir, inputfile2),
                        trim_quantiles=trim_quantiles,
                        n_poly=n_poly, plothist=False,
                        usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'A1_e', 'A2_e',
                                 'A3_e',
                                 'phi1', 'phi2', 'phi3', 'phi1_e', 'phi2_e', 'phi3_e', 'phi21', 'phi31', 'phi21_e',
                                 'phi31_e',
                                 'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift'],
                        input_feature_names=input_feature_names, y_col='feh', output_feature_indices=feature_indx,
                        yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'])

    y2 = y2 + df2['shift'].to_numpy()
    y = np.hstack((y, y2))
    yw = np.hstack((yw, yw2))
    X = np.vstack((X, X2))
    df = pd.concat((df, df2))
    df_orig = pd.concat((df_orig, df2_orig))

if inputfile3 is not None:
    X3, y3, yw3, df3, feature_names, df3_orig = \
        io.load_dataset(os.path.join(rootdir, inputfile3),
                        trim_quantiles=trim_quantiles,
                        n_poly=n_poly, plothist=False,
                        usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'A1_e', 'A2_e',
                                 'A3_e',
                                 'phi1', 'phi2', 'phi3', 'phi1_e', 'phi2_e', 'phi3_e', 'phi21', 'phi31', 'phi21_e',
                                 'phi31_e',
                                 'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift'],
                        input_feature_names=input_feature_names, y_col='feh', output_feature_indices=feature_indx,
                        yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'])

    y3 = y3 + df3['shift'].to_numpy()
    y = np.hstack((y, y3))
    yw = np.hstack((yw, yw3))
    X = np.vstack((X, X3))
    df = pd.concat((df, df3))
    df_orig = pd.concat((df_orig, df3_orig))

df['A1_e'] = df['A1_e'].replace(0.0, 0.001)
df['A2_e'] = df['A2_e'].replace(0.0, 0.001)
df['A3_e'] = df['A3_e'].replace(0.0, 0.001)
df['phi1_e'] = df['phi1_e'].replace(0.0, 0.001)
df['phi2_e'] = df['phi2_e'].replace(0.0, 0.001)
df['phi3_e'] = df['phi3_e'].replace(0.0, 0.001)
df['phi31_e'] = df['phi31_e'].replace(0.0, 0.001)
df['phi21_e'] = df['phi21_e'].replace(0.0, 0.001)

blazhko_mask = (df['type'] == 'RRab-BL') | (df['type'] == 'RRc-BL')

print('Input features:')
print(feature_names)

# ======================================================================================================================
# Perform initial linear regression:

reg = LinearRegression(fit_intercept=True)
reg.fit(X, y, sample_weight=yw)
r2score = reg.score(X, y.reshape(-1, 1))
yhat = reg.predict(X)
residual_stdev = np.std(y - yhat)
print("R^2 score = {0:.3f}".format(r2score))
print("residual st.dev. = {0:.3f}".format(residual_stdev))
io.plot_residual(y, yhat, xlabel="$[Fe/H]_{HR}$", ylabel="$[Fe/H]_{pred.}$",
                 plotrange=(-3, 0.5), fname=ols_residual_figure)
print("intercept = {0:.3f}".format(reg.intercept_))
for coef, fname in zip(reg.coef_, feature_names):
    print("coef_{0}:  {1:.4f}".format(fname, coef))

# ======================================================================================================================
# Define Bayesian model:

with pm.Model() as model:
    # largely uninformative prior on the linear function's parameters:
    icept = pm.Normal('icept', reg.intercept_, sigma=5)

    coefs = dict()
    for coef, fname in zip(reg.coef_, feature_names):
        coefs['coef_' + fname] = pm.Normal('coef_' + fname, coef, sigma=5)

    # informative priors on the latent regressors:
    a1_latent = pm.Normal('a1_latent', mu=df['A1'], sigma=df['A1_e'], shape=df['A1'].shape)
    a2_latent = pm.Normal('a2_latent', mu=df['A2'], sigma=df['A2_e'], shape=df['A2'].shape)

    phi1_latent = pm.Normal('phi1_latent', mu=df['phi1'], sigma=df['phi1_e'], shape=df['phi1'].shape)
    phi3_latent = pm.Normal('phi3_latent', mu=df['phi3'], sigma=df['phi3_e'], shape=df['phi3'].shape)

    phi31_latent = pm.Deterministic('phi31_latent', shift_phase(phi3_latent - 3 * phi1_latent, mean_31_phase))

    # likelihoods of the regressors:
    likelihood_phi1 = pm.Normal('lh_phi1', mu=phi1_latent, sigma=df['phi1_e'], observed=df['phi1'])
    likelihood_phi3 = pm.Normal('lh_phi3', mu=phi3_latent, sigma=df['phi3_e'], observed=df['phi3'])
    likelihood_a1 = pm.Normal('lh_A1', mu=a1_latent, sigma=df['A1_e'], observed=df['A1'])
    likelihood_a2 = pm.Normal('lh_A2', mu=a2_latent, sigma=df['A2_e'], observed=df['A2'])

    features = dict()
    features['period'] = df['period']
    features['A1'] = a1_latent
    features['A2'] = a2_latent
    features['phi31'] = phi31_latent

    y_latent = pm.Deterministic('y_latent', icept + coefs['coef_period'] * features['period'] +
                                coefs['coef_A1'] * features['A1'] +
                                coefs['coef_A2'] * features['A2'] +
                                coefs['coef_phi31'] * features['phi31'])

    likelihood_y = pm.Normal('lh_y', mu=y_latent, sigma=df['e_feh'], observed=y)


# Sample posterior:

tuningsteps = 10000

with model:
    trace = pm.sample(50000, tune=tuningsteps, cores=4, chains=4, init='advi', discard_tuned_samples=True,
                      return_inferencedata=True, target_accept=0.95)

with model:
    axes = az.plot_trace(trace, var_names=["icept", "coef"], filter_vars="like",
                         lines=tuple([(k, {}, [v['mean'], v['hdi_95%'], v['hdi_5%']])
                                      for k, v in
                                      az.summary(trace, var_names=["icept", "coef"], filter_vars="like",
                                                 hdi_prob=0.9).iterrows()]))

    fig = axes.ravel()[0].figure
    fig.savefig(traceplot_figure + '.' + figformat, format=figformat)

    summary = az.summary(trace, var_names=["icept", "coef"], filter_vars="like", kind='diagnostics')
    print(summary)
    summary = az.summary(trace, var_names=["icept", "coef"], filter_vars="like", kind='stats', hdi_prob=0.90)
    print(summary)
    summary = az.summary(trace, var_names=["icept", "coef"], filter_vars="like", kind='stats', hdi_prob=0.95)
    print(summary)

    axes = az.plot_pair(trace, var_names=["icept", "coef"], filter_vars="like", marginals=True,
                        kind='kde', figsize=(15, 15))

    fig = axes.ravel()[0].figure
    fig.get_axes()[0].set_ylabel("$\\theta_0$")
    fig.get_axes()[1].set_ylabel("$\\theta_P$")
    fig.get_axes()[3].set_ylabel("$\\theta_{A_1}$")
    fig.get_axes()[6].set_ylabel("$\\theta_{A_2}$")
    fig.get_axes()[10].set_ylabel("$\\theta_{\phi_{31}}$")

    fig.get_axes()[10].set_xlabel("$\\theta_0$")
    fig.get_axes()[11].set_xlabel("$\\theta_P$")
    fig.get_axes()[12].set_xlabel("$\\theta_{A_1}$")
    fig.get_axes()[13].set_xlabel("$\\theta_{A_2}$")
    fig.get_axes()[14].set_xlabel("$\\theta_{\phi_{31}}$")

    fig = axes.ravel()[0].figure
    fig.savefig(pairplot_figure + '.' + figformat, format=figformat)

params = dict()
params['icept'] = summary['mean'][0]
reg.intercept_ = summary['mean'][0]
for i, fname in enumerate(list(feature_names)):
    params[fname] = summary['mean'][i + 1]
    reg.coef_[i] = summary['mean'][i + 1]


yhat_mc = reg.predict(X)
residual_stdev = np.std(y - yhat_mc)
print("residual st.dev. = {0:.3f}".format(residual_stdev))
if mark_blazhko:
    io.plot_residual(y, yhat_mc, highlight=blazhko_mask, xlabel="$[Fe/H]_{HR}$", ylabel="$[Fe/H]_{pred.}$",
                     plotrange=(-3.2, -0.3), fname=residual_plot_figure + '.' + figformat, format=figformat)
else:
    io.plot_residual(y, yhat_mc, xlabel="$[Fe/H]_{HR}$", ylabel="$[Fe/H]_{pred.}$",
                     plotrange=(-3.2, -0.3), fname=residual_plot_figure + '.' + figformat, format=figformat)

print("intercept = {0:.3f}".format(reg.intercept_))

joblib.dump(reg, output_model_file)

icept_trace = trace['posterior'].icept.to_series().to_numpy()
coef_period_trace = trace['posterior'].coef_period.to_series().to_numpy()
coef_A1_trace = trace['posterior'].coef_A1.to_series().to_numpy()
coef_A2_trace = trace['posterior'].coef_A2.to_series().to_numpy()
# coef_phi21_trace = trace['posterior'].coef_phi21.to_series().to_numpy()
coef_phi31_trace = trace['posterior'].coef_phi31.to_series().to_numpy()

np.set_printoptions(precision=4, suppress=True)

covmat = np.cov(np.vstack((icept_trace, coef_period_trace, coef_A1_trace, coef_A2_trace,
                           coef_phi31_trace)), ddof=0)
print(covmat)

corrmat = np.corrcoef(np.vstack((icept_trace, coef_period_trace, coef_A1_trace, coef_A2_trace,
                                 coef_phi31_trace)))
print(corrmat)

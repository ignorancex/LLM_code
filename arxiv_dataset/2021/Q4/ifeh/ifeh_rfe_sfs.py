import os
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, GroupKFold
from sklearn.feature_selection import RFE, SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from ifeh_io import *
import seaborn as sns

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

# Output file:
outputfile = "ifeh_rfe_sfs1.6_RRab_pred.dat"

# PCA model (if None, no PCA transformation will be used):
# pca_model = "rrab_fourier_pca.sav"
# pca_model = "rrc_fourier_pca.sav"
pca_model = None

# If save_model is True, the predictive model will be saved to outputfile_model using joblib
save_model = True
outputfile_model = "ifeh_rfe_sfs1.6_RRab_model.sav"

# The name of the plot showing the predicted vs ground truth values.
residual_plot = "ifeh_rfe_sfs1.6_RRab_residual"
# Mark the Blazhko stars in the plot?
mark_blazhko = True

# ----------

# The names of the input features:
input_feature_names = ['period', 'A1', 'A2', 'A3', 'phi21', 'phi31']    # use this for standard Fourier representation
# input_feature_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']    # use this for PCA representation

# Threshold rejections:
subset_expr = '(type=="RRab" or type=="RRab-BL") and keep==1 and snr>95 and phcov>0.8'
# subset_expr = 'type=="RRab" and keep==1 and snr>95 and phcov>0.8'
# subset_expr = '(type=="RRc" or type=="RRc-BL") and keep==1 and snr>70 and phcov>0.8'
# subset_expr = 'type=="RRc" and keep==1 and snr>70 and phcov>0.8'

# Run feature selection algorithm?
select_features = True

# If select_features is False, and we already know the indices of the best feature set, it can be specified here:
feature_indices = [0, 2, 5]

# Bin input data into one point per object?
# This should be avoided since it will introduce bias to the input data set due to small number statistics.
bin_objects = False

# Cross-validation parameters:
n_repeats = 10       # number of repeats in the k-fold cross-validation
n_folds = 20         # number of folds in the k-fold cross-validation
is_groupcv = False   # Use group-based cross-validation?
                     #   (For the provided data sets, results are very similar in either case.)

# Filename for the boxplot showing the feature selection results:
boxplotname = "ifeh_rfe_sfs1.6_RRab_boxplot.pdf"
figformat = "pdf"

# Dictrionaries storing the figure labels for different feature sets.
# Uncomment the one that corresponding to the input feature set.
# n_poly=1, no PCA
labels_dict = {'period':"$P$", 'phi31':"$\phi_{31}$", 'A2':"$A_2$", 'A3':"$A_3$", 'A1':"$A_1$", 'phi21':"$\phi_{21}$"}

# n_poly=2, no PCA
# labels_dict = {'period':"$P$", 'A1':"$A_1$", 'A2':"$A_2$", 'A3':"$A_3$", 'phi21':"$\phi_{21}$", 'phi31':"$\phi_{31}$", \
#                'period^2':"$P^2$", 'period A1':"$PA_1$", 'period A2':"$PA_2$", 'period A3':"$PA_3$", \
#                'period phi21':"$P\phi_{21}$", 'period phi31':"$P\phi_{31}$", 'A1^2':"$A_1^2$", 'A1 A2':"$A_1A_2$", \
#                'A1 A3':"$A_1A_3$", 'A1 phi21':"A_1\phi_{21}", 'A1 phi31':"$A_1\phi_{31}$", \
#                'A2^2':"$A_2^2$", 'A2 A3':"$A_2A_3$", 'A2 phi21':"$A_2\phi_{21}$", 'A2 phi31':"$A_2\phi_{31}$", \
#                'A3^2':"$A_3^2$", 'A3 phi21':"$A_3\phi_{21}$", 'A3 phi31':"$A_3\phi_{31}$", \
#                'phi21^2':"$\phi_{21}^2$", 'phi21 phi31':"$\phi_{21}\phi_{31}$", 'phi31^2':"$\phi_{31}^2$"}

# n_poly=1, PCA
# labels_dict = {'E1':"$u_1$", 'E2':"$u_2$", 'E3':"$u_3$", 'E4':"$u_4$", 'E5':"$u_5$", 'E6':"$u_6$"}

# n_poly=2, PCA
# labels_dict = {'E1':"$u_1$", 'E2':"$u_2$", 'E3':"$u_3$", 'E4':"$u_4$", 'E5':"$u_5$", 'E6':"$u_6$", \
#                'E1^2':"$u_1^2$", 'E1 E2':"$u_1u_2$", 'E1 E3':"$u_1u_3$", 'E1 E4':"$u_1u_4$", 'E1 E5':"$u_1u_5$", \
#                'E1 E6':"$u_1u_6$", 'E2^2':"$u_2^2$", 'E2 E3':"$u_2u_3$", 'E2 E4':"$u_2u_4$", 'E2 E5':"$u_2u_5$", \
#                'E2 E6':"$u_2u_6$", 'E3^2':"$u_3^2$", 'E3 E4':"$u_3u_4$", 'E3 E5':"$u_3u_5$", 'E3 E6':"$u_3u_6$",\
#                'E4^2':"$u_4^2$", 'E4 E5':"$u_4u_5$", 'E4 E6':"$u_4u_6$", 'E5^2':"$u_5^2$", 'E5 E6':"$u_5u_6$", \
#                'E6^2':"$u_6^2$"}

# If trim_quantiles is provided, data beyond the lower and upper quantiles (qlo, qhi)
# of the listed column names will be trimmed.
trim_quantiles = None
qlo = 0.01  # lower percentile for data rejection
qhi = 0.99  # upper percentile for data rejection

is_test = False  # use an artificial dataset?
n_features = 8  # number of features in the artificial data set if is_test is True

# Standard scale the input data?
# In this case, this has only aesthetical purposes
# (the scaling before PCA transformation is integrated into the transformer object).
scale_data = True

# Feature selection parameters:
min_n_features = 1  # minimum number of features to try
max_n_features = 6  # maximum number of features to try
n_poly = 1  # maximum order of polynomial features
use_sfs = True  # if True, sklearn.SFS is used, if False, sklearn.RFE is used
is_backward = False    # do backward SFS, in addition to forward SFS?
scoring = 'neg_mean_absolute_error'    # cross-validation scorer
ylabel = 'CV MAE'

seed = 41    # Random seed for reproducibility.

# END OF PREAMBLE
# ======================================================================================================================


if pca_model is not None:
    # if a PCA transformation is used, the do not standard-scale the data, since it is already included in the
    # PCA transformer object.
    scale_data = False

# get a list of models to evaluate
def get_models(min_n, max_n, n_features, cv=None, scoring='neg_mean_absolute_error', scale_data=False):
    models = dict()
    if cv is not None:
        arr_loop = np.linspace(min_n, max_n, max_n - min_n + 1) / n_features
        for i, frac in enumerate(arr_loop):
            # print(i, frac)
            steps = list()
            if scale_data:
                steps.append(('scaler', StandardScaler()))
            steps.append(('selector', SFS(estimator=LinearRegression(fit_intercept=True),
                                          n_features_to_select=frac, direction='forward', scoring=scoring, cv=cv,
                                          n_jobs=-1)))
            steps.append(('estimator', LinearRegression(fit_intercept=True)))
            models[str(i + 1) + 'f'] = Pipeline(steps=steps)

            if is_backward:
                if i > 0 and frac < 1.0:  # forward and backward SFS is the same for the 1st and last model.
                    steps = list()
                    if scale_data:
                        steps.append(('scaler', StandardScaler()))
                    steps.append(('selector', SFS(estimator=LinearRegression(fit_intercept=True),
                                                  n_features_to_select=frac, direction='backward', scoring=scoring, cv=cv,
                                                  n_jobs=-1)))
                    steps.append(('estimator', LinearRegression(fit_intercept=True)))
                    models[str(i + 1) + 'b'] = Pipeline(steps=steps)

    else:
        for i in range(min_n, max_n + 1):
            steps = list()
            steps.append(('selector', RFE(estimator=DecisionTreeRegressor(),
                                          step=1, n_features_to_select=i)))
            steps.append(('estimator', LinearRegression(fit_intercept=True)))
            models[str(i)] = Pipeline(steps=steps)

    return models


# function to evaluate a model using cross-validation
def evaluate_model(model, X, y, yw, cv, scoring='neg_mean_absolute_error'):
    # cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=1)
    # scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise', fit_params={'estimator__sample_weight': yw})
    selector = model['selector']
    # selector.fit(X, y)
    selector.fit(X, y)
    # support_mask = selector.support_
    feature_indices = selector.get_support(indices=True)
    scores = cross_val_score(model['estimator'], X[:, feature_indices], y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise',
                             fit_params={'sample_weight': yw})

    return scores, feature_indices


# ----------------------------------------------------------------------------------------


if is_test:
    X, y, feature_names = load_test_dataset(n_features=8, n_poly=n_poly)
    yw = np.ones_like(y)
else:

    X, y, yw, df, feature_names, df_orig = \
        load_dataset(os.path.join(rootdir, inputfile1),
                     trim_quantiles=trim_quantiles, qlo=qlo, qhi=qhi,
                     n_poly=n_poly, plothist=True, histfig="hist_CHR",
                     usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'phi21', 'phi31',
                              'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift', 'oo_class'],
                     input_feature_names=input_feature_names, y_col='feh',
                     yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'], pca_model=pca_model)

    if inputfile2 is not None:

        X2, y2, yw2, df2, feature_names, df2_orig = \
            load_dataset(os.path.join(rootdir, inputfile2),
                         trim_quantiles=trim_quantiles, qlo=qlo, qhi=qhi,
                         n_poly=n_poly, plothist=True, histfig="hist_LIT1",
                         usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'phi21', 'phi31',
                                  'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift', 'oo_class'],
                         input_feature_names=input_feature_names, y_col='feh',
                         yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'], pca_model=pca_model)

        y2 = y2 + df2['shift'].to_numpy()
        y = np.hstack((y, y2))
        yw = np.hstack((yw, yw2))
        X = np.vstack((X, X2))
        df = pd.concat((df, df2))
        df_orig = pd.concat((df_orig, df2_orig))

    if inputfile3 is not None:

        X3, y3, yw3, df3, feature_names, df3_orig = \
            load_dataset(os.path.join(rootdir, inputfile3),
                         trim_quantiles=trim_quantiles, qlo=qlo, qhi=qhi,
                         n_poly=n_poly, plothist=True, histfig="hist_LIT2",
                         usecols=['name', 'use_id', 'source', 'period', 'totamp', 'A1', 'A2', 'A3', 'phi21', 'phi31',
                                  'phcov', 'snr', 'feh', 'e_feh', 'type', 'keep', 'shift', 'oo_class'],
                         input_feature_names=input_feature_names, y_col='feh',
                         yerr_col='e_feh', subset_expr=subset_expr, dropna_cols=['use_id'], pca_model=pca_model)

        y3 = y3 + df3['shift'].to_numpy()
        y = np.hstack((y, y3))
        yw = np.hstack((yw, yw3))
        X = np.vstack((X, X3))
        df = pd.concat((df, df3))
        df_orig = pd.concat((df_orig, df3_orig))

names = df['use_id'].to_numpy()

blazhko_mask = (df['type']=='RRab-BL') | (df['type']=='RRc-BL')

# Find the list of unique names and their numbers of occurrences
unique_names, name_counts = np.unique(names, return_counts=True)

singles = unique_names[name_counts == 1]    # names occurring only once
multiples = unique_names[name_counts > 1]   # names occurring more than once

n_objects = len(unique_names)               # number of (unique) objects
n_multiples = len(multiples)                # number of names occurring more than once
n_singles = len(singles)                    # number of names occurring only once


print("{} unique objects found.".format(n_objects))
print("{} objects have single and {} objects have multiple measurements.".format(n_singles, n_multiples))

if bin_objects:
    grouped = df.groupby('name', sort=False)
    # compute corrected std. dev. of feh per group (returns NaN if group has a single entry)
    ybin_err = grouped['feh'].agg(lambda x: np.nan if len(x) < 2 else np.sqrt(np.sum(np.abs(x - np.mean(x)) ** 2) / (len(x) - 1.5 + 1. / (8 * (len(x) - 1)))))
    # substitute NaN values with the original uncertainties
    feherr_bin = grouped['e_feh'].agg(lambda x: x.iloc[0])
    ybin_err[ybin_err.isna()] = feherr_bin[ybin_err.isna()]

    df['feh_shifted'] = df['feh'] + df['shift']
    # function for returning the weighted mean:
    wm = lambda x: np.average(x.feh_shifted, weights=x.weight)
    ybin = grouped.apply(wm)


    objnames = ybin_err.index.to_numpy()

    Xbin = grouped[input_feature_names].agg(lambda x: x.iloc[0])

    yw = 1.0 / ybin_err.to_numpy() ** 2
    y = ybin.to_numpy()
    X = Xbin.to_numpy()


ndata = len(y)

print("The dataset consists of {} individual data points.".format(ndata))

print('Input features:')

print(feature_names)
n_features = len(feature_names)

# define cross-validation splitter:
if is_groupcv  and  not bin_objects:
    groups = df.groupby(by='use_id', sort=False).ngroup().to_numpy()
    cv = list(GroupKFold(n_splits=n_folds).split(X, y, groups=groups))
else:
    groups = None
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)

if select_features:
    max_n_feat = np.min([max_n_features, X.shape[1]])
    # if max_n_features>X.shape[1]:
    #   max_n_feat = 1.0
    print('Maximum number of features to try: {}'.format(max_n_feat))

    # get the models to evaluate
    if use_sfs:
        models = get_models(min_n_features, max_n_feat, n_features, cv=cv, scoring=scoring)
    else:
        models = get_models(min_n_features, max_n_feat, n_features, cv=None, scoring=scoring)

    # evaluate the models and store results
    results, results_mean, names, nfeat, feature_indices, vnames = list(), list(), list(), list(), list(), list()

    # if is_groupcv:
    #     print("======================================")
    #     print("model  Nfeat  score  selected features")
    #     print("--------------------------------------")
    # else:
    print("=========================================================")
    print("model  Nfeat  mean score  stdev. score  selected features")
    print("---------------------------------------------------------")

    for name, model in models.items():
        scores, feature_indx = evaluate_model(model, X, y, yw, cv)
        results.append(-scores)
        nfeat.append(len(feature_indx))
        names.append(name)
        this_feature_subset = np.array(feature_names)[feature_indx]
        feature_indices.append(feature_indx)
        mean_score = np.mean(-scores)
        # if is_groupcv:
        #     print(
        #         '%s       %2d     %.4f        %s' % (name, len(feature_indx), mean_score, this_feature_subset))
        # else:
        results_mean.append(mean_score)
        print('%s       %2d     %.4f      %.4f        %s' % (
            name, len(feature_indx), mean_score, np.std(scores), np.array(feature_names)[feature_indx]))


    # ----------------------------------------------------------------------------
    # Create boxplot from the CV results:

    if not is_backward:
        # Get list of sequentially added features as x-labels for the boxplot:
        labels = list()

        for i, a in enumerate(feature_indices):
            if i > 0:
                c = np.setdiff1d(np.union1d(feature_indices[i - 1], a), np.intersect1d(feature_indices[i - 1], a))
                labels.append(np.array(feature_names)[c][0])
            else:
                labels.append(np.array(feature_names)[a][0])

        labels_nice = list()
        for l in labels:
            labels_nice.append(labels_dict[l])

    else:
        # use model names as x-labels for the boxplot:
        labels = names

    fig = plt.figure(figsize=(5, 4))
    plt.xlabel("sequentially added features")
    plt.ylabel(ylabel)
    plt.boxplot(results, labels=labels_nice, showmeans=True, whis=9999)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(rootdir, boxplotname), format=figformat)
    plt.close(fig)

    # ----------------------------------------------------------------------------

    # Select the best-performing model and print out its properties:
    best_model_index = np.argmin(results_mean)
    # best_n_features = int(names[best_model_index])
    best_n_features = nfeat[best_model_index]
    best_feature_indices = feature_indices[best_model_index]

else:

    best_feature_indices = feature_indices
    best_n_features = len(feature_indices)

print('Optimal number of features: {}'.format(best_n_features))

best_features = np.array(feature_names)[best_feature_indices]

print('Optimal set of features:')
print(list(best_features), sep=', ')
print('Indices of optimal features:')
print(list(best_feature_indices), sep=', ')

# fit entire dataset:

steps = list()
if scale_data:
    steps.append(('scaler', StandardScaler()))

steps.append(('estimator', LinearRegression(fit_intercept=True)))
final_model = Pipeline(steps=steps)


final_model.fit(X[:, best_feature_indices], y, estimator__sample_weight=yw)
if save_model:
    joblib.dump(final_model, outputfile_model)
# r2score = final_model.score(X[:, best_feature_indices], y, estimator__sample_weight=yw)
# print('R2 score (entire dev. set) = {0:.3f}'.format(r2score))

print('Fitted coefficients:')
print(final_model['estimator'].coef_)

print('Fitted intercept:')
print(final_model['estimator'].intercept_)

yhat = final_model.predict(X[:, best_feature_indices])

# Plot prediction vs ground truth:
if mark_blazhko:
    plot_residual(y, yhat, highlight=blazhko_mask, xlabel="$y$", ylabel="$\hat y_{ols/sfs}$",
                  fname=os.path.join(rootdir, residual_plot))
else:
    plot_residual(y, yhat, xlabel="$y$", ylabel="$\hat y_{ols/sfs}$", fname=os.path.join(rootdir, residual_plot))

# Compute and print various performance metrics:
stdev_fit = np.std(yhat-y)
print('Residual standard deviation = {0:.3f}'.format(stdev_fit))

scores = cross_val_score(final_model, X[:, best_feature_indices], y, scoring='neg_root_mean_squared_error',
                         cv=cv, n_jobs=-1, error_score='raise', fit_params={'estimator__sample_weight': yw})
print('WRMSE (CV) = {0:.4f} +/- {1:.4f}'.format(-np.mean(scores), np.std(scores)))

scores = cross_val_score(final_model, X[:, best_feature_indices], y, scoring='neg_root_mean_squared_error',
                         cv=cv, n_jobs=-1, error_score='raise')
print('RMSE (CV) = {0:.4f} +/- {1:.4f}'.format(-np.mean(scores), np.std(scores)))

scores = cross_val_score(final_model, X[:, best_feature_indices], y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1, error_score='raise',
                         fit_params={'estimator__sample_weight': yw})
print('WMAE (CV) = {0:.4f} +/- {1:.4f}'.format(-np.mean(scores), np.std(scores)))

scores = cross_val_score(final_model, X[:, best_feature_indices], y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1, error_score='raise')
print('MAE (CV) = {0:.4f} +/- {1:.4f}'.format(-np.mean(scores), np.std(scores)))


r2score = r2_score(y, yhat, sample_weight=yw)
print('R2 score (all) = {0:.4f}'.format(r2score))


if not is_test:
    # Save predictions to file for the entire development set:
    # outarr = np.rec.fromarrays((df['name'].to_numpy(), df['source'].to_numpy(), df['oo_class'].to_numpy(), y, yhat))
    if bin_objects:
        outarr = np.rec.fromarrays((objnames, y, yhat))
        np.savetxt(os.path.join(rootdir, outputfile), outarr, fmt='%s   %.4f  %.4f')
    else:
        outarr = np.rec.fromarrays((df['name'].to_numpy(), df['source'].to_numpy(), y, yhat))
        np.savetxt(os.path.join(rootdir, outputfile), outarr, fmt='%s   %s  %.4f  %.4f')

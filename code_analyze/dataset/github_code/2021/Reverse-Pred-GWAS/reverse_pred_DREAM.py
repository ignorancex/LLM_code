import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import seaborn as sns
from scipy import stats
from scipy.special import betainc
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
import statsmodels.formula.api as smf
import re

geno_file = 'DREAM5_SysGenA999/DREAM5_SysGenA999_Network1_Genotype.tsv'
expr_file = 'DREAM5_SysGenA999/DREAM5_SysGenA999_Network1_Expression.tsv'
gold_file = 'DREAM5_SysGenA_GoldStandard/A999/DREAM5_SysGenA999_Edges_Network1.tsv'

df_geno = pd.read_table(geno_file, usecols=range(1000))
df_expr = pd.read_table(expr_file, usecols=range(1000))
temp_df = pd.read_table(gold_file, header=None)
temp_df[0] = temp_df[0].str.strip('G').astype(int)
temp_df[1] = temp_df[1].str.strip('G').astype(int)

df_gold = pd.crosstab(temp_df[0], temp_df[1])
idx = df_gold.columns.union(df_gold.index).astype(int)
df_gold = df_gold.reindex(index = idx, columns=idx, fill_value=0).T


print('Genetype Data Shape',df_geno.shape)
print('Expression Data Shape',df_expr.shape)
print('Ground Truth Data Shape',df_gold.shape)

# Expression values normalized to have unit standard deviation and zero mean for each feature
X_p = StandardScaler().fit_transform(df_expr.to_numpy())

kf = KFold(n_splits=5)

idx = []
for i in tqdm(range(df_geno.shape[1])):
    if df_gold.values[:,i].sum() == 0:
        idx.append(i)

def corrcoef(matrix):
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p
# df_geno.shape[1]

IMP_rf = np.zeros([df_geno.shape[1], 1000, 5])
zero_class = np.zeros([df_geno.shape[1], 5])
SCORES_rf = np.zeros([df_geno.shape[1], 5])
AUROC_rf = np.zeros([df_geno.shape[1], 5])

IMP_ridge = np.zeros([df_geno.shape[1], 1000, 5])
SCORES_ridge = np.zeros([df_geno.shape[1], 5])
AUROC_ridge = np.zeros([df_geno.shape[1], 5])

IMP_svm = np.zeros([df_geno.shape[1], 1000, 5])
SCORES_svm = np.zeros([df_geno.shape[1], 5])
AUROC_svm = np.zeros([df_geno.shape[1], 5])

IMP_svm = np.zeros([df_geno.shape[1], 1000, 5])
SCORES_svm = np.zeros([df_geno.shape[1], 5])
AUROC_svm = np.zeros([df_geno.shape[1], 5])


fold = 0
SCORES_nb = np.zeros([df_geno.shape[1], 5])
IMP_corr = np.zeros([df_geno.shape[1],1000, 5])
AUROC_corr = np.zeros([df_geno.shape[1], 5])
p_vals = np.zeros([df_geno.shape[1],1000, 5])
for train_index, test_index in kf.split(X_p):
    
    print("Processing fold:%d"%(fold+1))
    X_train, y_train = X_p[train_index], df_geno.values[train_index]
    X_test, y_test = X_p[test_index], df_geno.values[test_index]

    for i in tqdm(range(df_geno.shape[1])):
        ytrain = y_train[:,i]
        ytest = y_test[:,i]
        new_gold = df_gold.values[:,i]

        if new_gold.sum() == 0:
            continue
        Xtrain = X_train.copy()
        Xtest = X_test.copy()
        
        ## RANDOM FOREST ####################
        rfr = RandomForestRegressor(random_state=42, n_jobs=-1).fit(Xtrain,ytrain)
        imp = rfr.feature_importances_
        
        IMP_rf[i, :, fold] = imp
        
        zero_class[i, fold] = (y_test.shape[0] - ytest.sum())
        
        SCORES_rf[i, fold] = mean_squared_error(ytest,rfr.predict(Xtest), squared=False)
        
        ## RIDGE REGRESSION #################
        ridge = linear_model.Ridge(alpha = 100).fit(Xtrain,ytrain)
        ridge_coefs = ridge.coef_
        IMP_ridge[i, :, fold] = ridge_coefs      
        SCORES_ridge[i, fold] = mean_squared_error(ytest, ridge.predict(Xtest), squared=False)
        
        ## SUPPORT VECTOR ##################
        svm = SVR(kernel='linear').fit(Xtrain,ytrain)
        svm_coefs = svm.coef_
        IMP_svm[i, :, fold] = svm_coefs
        SCORES_svm[i, fold] = mean_squared_error(ytest, svm.predict(Xtest), squared=False)
        
        ### CORRELATION ######################
        matrix = np.concatenate((Xtrain,ytrain.reshape(Xtrain.shape[0],1)),axis=1)
        r1, p1 = corrcoef(matrix.T)
        p_vals[i, :, fold] = (p1[:-1,-1])
        corr_vals = r1[:-1,-1]
        IMP_corr[i,:, fold] = corr_vals
        
        ## NaiveBaiyes ###############################
        nb = GaussianNB().fit(Xtrain,ytrain)
        SCORES_nb[i, fold] = mean_squared_error(ytest, nb.predict(Xtest), squared=False) #nb.score(Xtest,ytest)
            
        AUROC_rf[i, fold] = roc_auc_score(new_gold, np.abs(imp))
        AUROC_ridge[i, fold] = roc_auc_score(new_gold, np.abs(ridge_coefs))
        AUROC_svm[i, fold] = roc_auc_score(new_gold, np.abs(svm_coefs).ravel())
        AUROC_corr[i, fold] = roc_auc_score(new_gold, np.abs(corr_vals))
    fold += 1
    
true_targets0 = df_gold.values.sum(axis=0)
true_targets1 = df_gold.values.sum(axis=1)

np.savez('dream_data_network1_reg_v2.npz',AUROC_corr=AUROC_corr, AUROC_rf=AUROC_rf, AUROC_ridge = AUROC_ridge,
         AUROC_svm = AUROC_svm, SCORES_rf=SCORES_rf, SCORES_ridge=SCORES_ridge, SCORES_svm=SCORES_svm,          
          SCORES_nb = SCORES_nb, idx=idx, p_vals=p_vals, 
        zero_class=zero_class, true_targets0=true_targets0, true_targets1=true_targets1,
        IMP_rf = IMP_rf, IMP_corr=IMP_corr, IMP_ridge = IMP_ridge, IMP_svm = IMP_svm)

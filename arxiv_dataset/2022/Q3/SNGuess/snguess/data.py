import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
# from ampel.ztf.util.ZTFIdMapper import to_ztf_id

def add_one(number):
    return number + 1

def data_cuts(training_data, nbr_det=2):
    
    # Might want to redo this for different number of detections
    # nbr_det = 2, 3, 4, 5, 6, 100
    nbr_det = 2
    if nbr_det<=2:
        max_duration = 3.5
        max_predetect = 3.5
        min_detmag = 16 # Things starting brighter than this must be stars
    elif nbr_det<=4:
        max_duration = 6.5
        max_predetect = 3.5
        min_detmag = 16 # Things starting brighter than this must be stars
    elif nbr_det<=6:
        max_duration = 10
        max_predetect = 3.5
        min_detmag = 16 # Things starting brighter than this must be stars
    elif nbr_det==100:
        # This is a special case where we look for a full lightcurve
        max_duration = 90
        max_predetect = 10
        min_detmag = 16 # Things starting brighter than this must be stars

    features = training_data.drop( np.where(training_data['bool_hasgaps'])[0])

    iTime = np.where(features["t_lc"]<max_duration)[0]
    features = features.iloc[iTime]

    iPre = np.where(features["t_predetect"]<=max_predetect)[0]
    features = features.iloc[iPre]

    # Skip cases with intervening upper limits for now
    if nbr_det<100:
        features = features[ features['bool_pure']]

    # Also, lets skip the few cases with duplicated pps
    iDup = np.where(features["cut_pp"]>0)[0]
    dupTransients = features.iloc[iDup]['snname'].unique()
    features = features[ ~features['snname'].isin(dupTransients) ]

    iMag = np.where(features["mag_det"]<min_detmag)[0]
    mTransients = features.iloc[iMag]['snname'].unique()
    features = features[ ~features['snname'].isin(mTransients) ]

    if nbr_det==100:
        iDet = np.where( (features["ndet"]<100) & (features["ndet"]>6) )[0]
    else:
        iDet = np.where(features["ndet"]==nbr_det)[0]
    features = features.iloc[iDet]

    return features

def merge_test_features_labels(test_data_features, bts_transients):
    
    # test_data_features['ztfname'] = to_ztf_id(test_data_features['stockid'])

    targeted_by_bts = pd.Series(True, index=bts_transients.index, name='targeted_by_bts')
    snguess = test_data_features.merge(targeted_by_bts, how='left', left_on='ztfname', right_index=True).fillna(value={'targeted_by_bts':False})

    snguess = pd.merge(snguess, bts_transients['type'], left_on=['ztfname'], how='left', right_index=True)

    snguess.loc[snguess.SNGuessBool.isna(),'SNGuess'] = 0.0

    snguess['SNGuessProb'] = 1 / (1 + np.exp(-snguess.SNGuess))

    return snguess

def get_training_input(training_data):

    # feat_name = 'distnr_med'
    feat_name = 'rcf_sn'

    training_data = training_data[(training_data[feat_name].notna() & training_data['targeted_by_bts'].notna())]

    X_train = training_data[feat_name].to_numpy().reshape(-1,1)
    y_train = training_data['targeted_by_bts'].to_numpy()

    return X_train, y_train

def get_testing_input(test_data):

    X_train = test_data['distnr_med'].to_numpy().reshape(-1,1)
    y_train = test_data['targeted_by_bts']

    return X_train, y_train

def plot_roc_curve(logreg, X_test, y_true):

    y_score = logreg.predict_proba(X_test)[:,1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(dpi=200, figsize=(4,4))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,          # AUC: reflects performance from plot with a single score
    )
    plt.plot([0, 1], [0, 1],                                 # Baseline performance (flipping a coin)
         color="navy", 
         lw=lw, 
         linestyle="--", 
         label='Baseline')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    return fpr, tpr, roc_auc
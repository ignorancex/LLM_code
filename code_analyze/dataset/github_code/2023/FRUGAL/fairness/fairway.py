import pandas as pd
import random,time
import numpy as np
import math,copy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pdb
from result.measure import calculate_recall,calculate_far,calculate_average_odds_difference, calculate_equal_opportunity_difference, get_counts, measure_final_score
from optimizer.flash import flash_fair_LSR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def adult_pp(df):
    ## Drop categorical features
    dataset_orig = df.drop(
        ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'capital-gain',
         'capital-loss', 'hours-per-week', 'native-country'], axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)

    ## Discretize age
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 60) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 50) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 40) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 30) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 20) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 10) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
    dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])

    ## Discretize education-num
    dataset_orig['education-num'] = np.where(dataset_orig['education-num'] > 12, 13, dataset_orig['education-num'])
    dataset_orig['education-num'] = np.where(dataset_orig['education-num'] <= 6, 5, dataset_orig['education-num'])
    return dataset_orig

def bank_pp(df):
    ## Drop categorical features
    dataset_orig = df.drop(['job', 'marital', 'default',
                          'housing', 'loan', 'contact', 'month', 'day',
                          'poutcome'], axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()
    # mean = dataset_orig.loc[:,"age"].mean()
    # dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)

    ## Chaneg symbolic to numeric column
    gle = LabelEncoder()
    genre_labels = gle.fit_transform(dataset_orig['education'])
    genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    dataset_orig['education'] = genre_labels

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    return dataset_orig

def compass_pp(df):
    ## Drop categorical features
    ## Removed two duplicate coumns - 'decile_score','priors_count'
    dataset_orig = df.drop(
        ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age', 'juv_fel_count', 'decile_score',
         'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
         'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
         'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
         'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
         'type_of_assessment', 'decile_score', 'score_text', 'screening_date', 'v_type_of_assessment', 'v_decile_score',
         'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'event'], axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
    dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
    dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1) & (dataset_orig['priors_count'] <= 3),
                                            3, dataset_orig['priors_count'])
    dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45', int(45), dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', int(25), dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', int(0), dataset_orig['age_cat'])
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
    ## Rename class column
    dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
    dataset_orig['age_cat'] = dataset_orig['age_cat'].astype(int)
    return dataset_orig


def default_pp(df):
    ## Change column values

    df['sex'] = np.where(df['sex'] == 2, 0, 1)

    ## Drop NULL values
    dataset_orig = df.dropna()

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    return dataset_orig


def hearthealth_pp(df):
    ## Drop NULL values
    dataset_orig = df.dropna()

    ## calculate mean of age column
    mean = dataset_orig.loc[:, "age"].mean()
    dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)

    ## Make goal column binary
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 0, 1, 0)

    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    return dataset_orig


def students_pp(df):
    ## Drop NULL values
    dataset_orig = df.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop(['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'],
                                     axis=1)

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
    dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
    dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
    dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
    dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
    dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
    dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
    dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
    dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 12, 1, 0)

    protected_attribute = 'sex'
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)
    return dataset_orig


def meps15_pp(df):
    # ## Drop NULL values
    MEPS15 = df.dropna()
    MEPS15 = MEPS15.rename(
        columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                 'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                 'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                 'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                 'POVCAT15': 'POVCAT', 'INSCOV15': 'INSCOV'})

    MEPS15 = MEPS15[MEPS15['PANEL'] == 19]
    MEPS15 = MEPS15[MEPS15['REGION'] >= 0]  # remove values -1
    MEPS15 = MEPS15[MEPS15['AGE'] >= 0]  # remove values -1
    MEPS15 = MEPS15[MEPS15['MARRY'] >= 0]  # remove values -1, -7, -8, -9
    MEPS15 = MEPS15[MEPS15['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
    MEPS15 = MEPS15[
        (MEPS15[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

    # ## Change symbolics to numerics
    MEPS15['RACEV2X'] = np.where((MEPS15['HISPANX'] == 2) & (MEPS15['RACEV2X'] == 1), 1, MEPS15['RACEV2X'])
    MEPS15['RACEV2X'] = np.where(MEPS15['RACEV2X'] != 1, 0, MEPS15['RACEV2X'])
    MEPS15 = MEPS15.rename(columns={"RACEV2X": "RACE"})

    # MEPS15['UTILIZATION'] = np.where(MEPS15['UTILIZATION'] >= 10, 1, 0)

    def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    MEPS15['TOTEXP15'] = MEPS15.apply(lambda row: utilization(row), axis=1)
    lessE = MEPS15['TOTEXP15'] < 10.0
    MEPS15.loc[lessE, 'TOTEXP15'] = 0.0
    moreE = MEPS15['TOTEXP15'] >= 10.0
    MEPS15.loc[moreE, 'TOTEXP15'] = 1.0

    MEPS15 = MEPS15.rename(columns={'TOTEXP15': 'UTILIZATION'})

    MEPS15 = MEPS15[['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                     'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'PERWT15F', 'UTILIZATION']]

    MEPS15 = MEPS15.rename(columns={"UTILIZATION": "Probability", "RACE": "race"})
    return MEPS15

def german_pp(df):
    ## Drop categorical features
    dataset_orig = df.drop(
        ['1', '2', '4', '5', '8', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20'], axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1,
                                              dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1,
                                              dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1,
                                              dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2,
                                              dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3,
                                              dataset_orig['credit_history'])

    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])

    ## ADD Columns
    dataset_orig['credit_history=Delay'] = 0
    dataset_orig['credit_history=None/Paid'] = 0
    dataset_orig['credit_history=Other'] = 0

    dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1,
                                                    dataset_orig['credit_history=Delay'])
    dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1,
                                                        dataset_orig['credit_history=None/Paid'])
    dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1,
                                                    dataset_orig['credit_history=Other'])

    dataset_orig['savings=500+'] = 0
    dataset_orig['savings=<500'] = 0
    dataset_orig['savings=Unknown/None'] = 0

    dataset_orig['savings=500+'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings=500+'])
    dataset_orig['savings=<500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings=<500'])
    dataset_orig['savings=Unknown/None'] = np.where(dataset_orig['savings'] == 3, 1,
                                                    dataset_orig['savings=Unknown/None'])

    dataset_orig['employment=1-4 years'] = 0
    dataset_orig['employment=4+ years'] = 0
    dataset_orig['employment=Unemployed'] = 0

    dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1,
                                                    dataset_orig['employment=1-4 years'])
    dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1,
                                                   dataset_orig['employment=4+ years'])
    dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1,
                                                     dataset_orig['employment=Unemployed'])
    dataset_orig['sex'] = dataset_orig['sex'].astype(int)
    dataset_orig_labels = dataset_orig['Probability'].values
    dataset_orig_labels = np.where(dataset_orig_labels == 1, 0, dataset_orig_labels)
    dataset_orig_labels = np.where(dataset_orig_labels == 2, 1, dataset_orig_labels)

    dataset_orig = dataset_orig.drop(['credit_history', 'savings', 'employment', 'Probability'], axis=1)
    dataset_orig['Probability'] = dataset_orig_labels
    return dataset_orig


def meps16_pp(df):
    # ## Drop NULL values
    MEPS16 = df.dropna()

    MEPS16 = MEPS16.rename(
        columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                 'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                 'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                 'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                 'POVCAT16': 'POVCAT', 'INSCOV16': 'INSCOV'})

    MEPS16 = MEPS16[MEPS16['PANEL'] == 21]
    MEPS16 = MEPS16[MEPS16['REGION'] >= 0]  # remove values -1
    MEPS16 = MEPS16[MEPS16['AGE'] >= 0]  # remove values -1
    MEPS16 = MEPS16[MEPS16['MARRY'] >= 0]  # remove values -1, -7, -8, -9
    MEPS16 = MEPS16[MEPS16['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
    MEPS16 = MEPS16[
        (MEPS16[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

    # ## Change symbolics to numerics
    MEPS16['RACEV2X'] = np.where((MEPS16['HISPANX'] == 2) & (MEPS16['RACEV2X'] == 1), 1, MEPS16['RACEV2X'])
    MEPS16['RACEV2X'] = np.where(MEPS16['RACEV2X'] != 1, 0, MEPS16['RACEV2X'])
    MEPS16 = MEPS16.rename(columns={"RACEV2X": "RACE"})

    # MEPS16['UTILIZATION'] = np.where(MEPS16['UTILIZATION'] >= 10, 1, 0)

    def utilization(row):
        return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

    MEPS16['TOTEXP16'] = MEPS16.apply(lambda row: utilization(row), axis=1)
    lessE = MEPS16['TOTEXP16'] < 10.0
    MEPS16.loc[lessE, 'TOTEXP16'] = 0.0
    moreE = MEPS16['TOTEXP16'] >= 10.0
    MEPS16.loc[moreE, 'TOTEXP16'] = 1.0

    MEPS16 = MEPS16.rename(columns={'TOTEXP16': 'UTILIZATION'})

    MEPS16 = MEPS16[['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                     'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'PERWT16F', 'UTILIZATION']]

    MEPS16 = MEPS16.rename(columns={"UTILIZATION": "Probability", "RACE": "race"})

    return MEPS16


def run_ten_times_default(dataset_orig, df_name):
    print(" ---------- Default Results --------")
    for i in range(3):
        print("----Run No----",i)
        start = time.time()
        ## Divide into train,validation,test
        dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3)
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        #### DEFAULT Learners ####
        # --- LSR
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR Default Config
        # --- CART
        # clf = tree.DecisionTreeClassifier(criterion="gini",splitter="best",min_samples_leaf=1,min_samples_split=2) # CART Default Config
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cnf_matrix_test = confusion_matrix(y_test, y_pred)

        print(cnf_matrix_test)

        TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

        print("recall:", 1 - calculate_recall(TP,FP,FN,TN))
        print("far:",calculate_far(TP,FP,FN,TN))
        print("aod for sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
        print("eod for sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
        print("aod for race:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'aod'))
        print("eod for race:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'eod'))
        end = time.time()
        print(end - start)

def run_ten_times_FLASH(dataset_orig, df_name):
    print(" ---------- FLASH Results --------")
    for i in range(3):
        print("----Run No----",i)
        start = time.time()
        ## Divide into train,validation,test
        dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3)
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


        # tuner = LR_TUNER()
        # best_config = tune_with_flash(tuner,  X_train, y_train, X_valid, y_valid, 'adult', dataset_orig_valid, 'sex')
        best_config = flash_fair_LSR(dataset_orig,"race","ABCD")
        print("best_config",best_config)
        p1 = best_config[0]
        if best_config[1] == 1:
            p2 = 'l1'
        else:
            p2 = 'l2'
        if best_config[2] == 1:
            p3 = 'liblinear'
        else:
            p3 = 'saga'
        p4 = best_config[3]
        clf = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
        # clf = tuner.get_clf(best_config)
        print("recall :", 1 - measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'far'))
        print("aod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'aod'))
        print("eod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'eod'))
        end = time.time()
        print(end - start)

def load_csv():
    path = 'dataset/'
    projects = ['adult.data', 'compas-scores-two-years', 'GermanData', 'default_of_credit_card_clients_first_row_removed', 'processed.cleveland.data', 'bank', 'Student', 'MEPS15', 'MEPS16']
    # projects = ['MEPS15', 'MEPS16', 'Students']
    # projects = ["GermanData", 'MEPS16']
    final_data = {}
    for p in projects:
        df = pd.read_csv(f'{path}{p}.csv')
        if p == 'adult.data':
            p_name = 'adult'
            attributes = ['sex', 'race']
            pp_df = adult_pp(df)
        elif p == 'bank':
            p_name = 'bank'
            attributes = ['age']
            pp_df = bank_pp(df)
        elif p == 'compas-scores-two-years':
            p_name = 'compas'
            attributes = ['sex', 'race']
            pp_df = compass_pp(df)
        elif p == 'processed.cleveland.data':
            p_name = 'hearthealth'
            attributes = ['age']
            pp_df = hearthealth_pp(df)
        elif p == 'MEPS15':
            p_name = 'MEPS15'
            attributes = ['race']
            pp_df = meps15_pp(df)
        elif p == 'MEPS16':
            p_name = 'MEPS16'
            attributes = ['race']
            pp_df = meps16_pp(df)
        elif p == 'Student':
            p_name = 'student'
            attributes = ['sex']
            pp_df = students_pp(df)
        elif p == 'GermanData':
            p_name = 'german'
            attributes = ['sex']
            pp_df = german_pp(df)
        else:
            attributes = ['sex']
            p_name = 'defaultcredit'
            pp_df = default_pp(df)

        pp_df.rename({'Probability': 'Label'}, axis=1, inplace=True)
        for a in attributes:
            final_data[f'{p_name}_{a}'] = pp_df
    return final_data
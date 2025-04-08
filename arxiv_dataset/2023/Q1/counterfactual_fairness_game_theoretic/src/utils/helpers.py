"""
This module containts helper functions to load data and get meta deta.
"""
import numpy as np
import pandas as pd
import logging
import yaml
import sys
from dfencoder.scalers import GaussRankScaler
from sklearn import preprocessing


def features_setting(data):
    """

    :param data:
    :type data:
    :return:
    :rtype:
    """
    dict_ = {}
    if data == "law":
        dict_['sensitive_features'] = ['race', 'sex']
        dict_['normal_features'] = ['LSAT', 'UGPA']
        dict_['categorical_features'] = ['race', 'sex']
        dict_['continuous_features'] = ['LSAT', 'UGPA']
        dict_['discrete_features'] = []
        dict_['target'] = 'ZFYA'
        dict_['full_features'] = ['race', 'sex', 'LSAT', 'UGPA']

    elif data == "adult":
        dict_['categorical_features'] = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
        dict_['continuous_features']  = ['age', 'hours_per_week']
        dict_['sensitive_features'] = ['race', 'gender']
        dict_['target'] = 'two_year_recid'
        dict_['full_features'] = dict_['categorical_features'] + dict_['continuous_features']
        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]
        dict_['target'] = 'income'
        dict_['discrete_features'] = []

    elif data == "compas":
        dict_['categorical_features'] = ['age_cat', 'c_charge_degree', 'sex', 'race']
        dict_['continuous_features'] = ['age']
        dict_['discrete_features'] = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        dict_['full_features'] = dict_['categorical_features'] + dict_['continuous_features'] + dict_['discrete_features']
        dict_['sensitive_features'] = ['race', 'sex']
        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]
        dict_['target'] = 'two_year_recid'


    elif data == "german":
        dict_['categorical_features'] = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
        dict_['continuous_features']  = ['age', 'hours_per_week']
        dict_['full_features'] = dict_['categorical_features'] + dict_['continuous_features']
        dict_['sensitive_features'] = ['race', 'gender']
        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]
        dict_['target'] = 'income'

    elif data == "bank":
        dict_['categorical_features'] = ['job', 'education', 'housing', 'loan', 'poutcome', 'marital']
        dict_['continuous_features']  = ['age', 'balance', 'duration', 'previous']
        dict_['full_features'] = dict_['categorical_features'] + dict_['continuous_features']
        dict_['sensitive_features'] = ['marital']
        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]
        dict_['target'] = 'deposit'
    return dict_

def preprocess_dataset(df, continuous_features, categorical_features):
    """ Normalize the continuous features, convert the categorical features to label encoder

    :param df:
    :type df:
    :param continuous_features:
    :type continuous_features:
    :param categorical_features:
    :type categorical_features:
    :return:
    :rtype:
    """
    # scaler = GaussRankScaler()
    # scaler.fit_transform(x)

    # for c in continuous_features:
    #     df[c] = (df[c] - df[c].mean()) / df[c].std()

    for c in categorical_features:
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])
        df[c] = pd.Categorical(df[c].values)
        del le

    return df

def load_adult_income_dataset(path, result_path, save_intermediate=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    adult_data = pd.read_csv(path, delimiter=", ", engine='python')

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status',
                                            'hours-per-week': 'hours_per_week',
                                            'sex': 'gender'})

    adult_data[['age', 'hours_per_week']] = adult_data[['age', 'hours_per_week']].astype(int)
    adult_data['income'] = adult_data['income'].astype(str)

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace(
        {'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov': 'Government'}})
    adult_data = adult_data.replace(
        {'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                                    'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                                                    'Handlers-cleaners': 'Blue-Collar',
                                                    'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                                                    'Priv-house-serv': 'Service',
                                                    'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                                                    'Tech-support': 'Service',
                                                    'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                                                    'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'}})

    adult_data = adult_data.replace({'marital_status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital_status', 'occupation', 'race', 'gender',
                             'hours_per_week', 'income']]

    adult_data['income'] = np.where(adult_data.income == '<=50K', 0, 1)

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School',
                                                   '12th': 'School', '5th-6th': 'School', '1st-4th': 'School',
                                                   'Preschool': 'School'}})

    if save_intermediate:
        adult_data.to_csv(result_path, index=False)

    return adult_data

def setup_logging(log_path):
    """

    :param log_path:
    :type log_path:
    :return:
    :rtype:
    """

    with open(log_path, 'w'):
        pass
    logger = logging.getLogger('CFairness')
    file_handler = logging.FileHandler(filename=log_path)
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(name)-12s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def load_config(config_path):
    """

    :param config_path:
    :type config_path:
    :return:
    :rtype:
    """
    with open(config_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf

# def get_predict(ae_model, generator, df, normal_features, full_features, l = ''):
#
#     GD_prediction = 'GD_prediction' + l
#
#     df_generator = df[normal_features].copy()
#     df_autoencoder = df[full_features].copy()
#
#     Z = ae_model.get_representation(df_autoencoder)
#     Z = Z.cpu().detach().numpy()
#     reg = LogisticRegression(solver='saga', max_iter=1000)
#     reg.fit(Z, df['ZFYA'].values)
#     y_pred = reg.predict(Z)
#     df["AL_prediction"] = y_pred
#
#     """Generator + Linear regression"""
#     Z = generator.custom_forward(df_generator)
#     Z = Z.cpu().detach().numpy()
#     reg = LogisticRegression(solver='saga', max_iter=1000)
#     reg.fit(Z, df['ZFYA'].values)
#     y_pred = reg.predict(Z)
#     df["GL_prediction"] = y_pred
#
#     """Generator + Discriminator"""
#     Z = generator.custom_forward(df_generator)
#     predictor_agnostic = discriminator_agnostic(Z)
#     y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
#     df[GD_prediction] = y_pred
#
#     return df

if __name__ == "__main__":
    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    df = load_adult_income_dataset(conf['data_adult'], conf['processed_data_adult'], True)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
import argparse
import logging
import sys
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import GerryFairClassifier


class CompasDataset(StandardDataset):


    def __init__(self, df, label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[['Female'], ['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=['age_cat', 'c_charge_degree',
                     'c_charge_desc'],
                 features_to_keep=['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=None,
                 metadata=None):

        super(CompasDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=None, metadata=metadata)

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='compas')
    # parser.add_argument('--data_name', type=str, default='adult')
    # parser.add_argument('--data_name', type=str, default='bank')

    args = parser.parse_args()
    dataname = args.data_name

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['evaluate_law'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)


    log_path = conf['log_train_{}'.format(dataname)]
    data_path = conf['data_{}'.format(dataname)]

    """Set up logging"""
    logger = setup_logging(log_path)

    """Load data"""
    df = pd.read_csv(data_path)

    """Setup features"""
    dict_ = features_setting(dataname)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    """Preprocess data"""
    for c in continuous_features:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    count_dict = {}
    for c in categorical_features:
        count_dict[c] = len(df[c].unique())
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)

    """Split train-test"""
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=0)

    """Setup dataset for AIF"""
    privileged_groups = [{'race': 0}]
    unprivileged_groups = [{'race': [1]}]
    all_privileged_classes = {"race": [1]}
    privileged_classes = [[1]]

    dataset_orig_train = CompasDataset(
        df_train,
        label_name=target,
        favorable_classes=[0],
        protected_attribute_names=sensitive_features,
        privileged_classes=privileged_classes,
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=full_features,
        na_values=[],
        metadata=None)

    dataset_orig_test = CompasDataset(
        df_test,
        label_name=target,
        favorable_classes=[0],
        protected_attribute_names=sensitive_features,
        privileged_classes=privileged_classes,
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=full_features,
        na_values=[],
        metadata=None)

    """Meta-fair classifier"""
    biased_model = MetaFairClassifier(tau=0, sensitive_attr="race", type="fdr").fit(dataset_orig_train)
    dataset_bias_test = biased_model.predict(dataset_orig_test)
    df_test['MetaFair'] = dataset_bias_test.labels.reshape(-1)

    """GerryFair Classifier"""
    C = 100
    print_flag = True
    gamma = .005
    max_iterations = 500
    fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                                     max_iters=max_iterations, heatmapflag=False)
    fair_model.fit(dataset_orig_train, early_termination=True)
    dataset_yhat = fair_model.predict(dataset_orig_test, threshold=False)
    y_prediction = np.where(dataset_yhat.labels.reshape(-1) >= 0.5, 1,0)
    df_test['GerryFair'] = dataset_yhat.labels.reshape(-1)
    df_test.to_csv(conf['result_aif_{}'.format(dataname)])




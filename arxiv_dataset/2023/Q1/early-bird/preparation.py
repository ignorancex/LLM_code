import numpy as np
import sys
from sklearn import preprocessing



def standardization_SVL(train_data, test_data, scale_name):


    if scale_name == "0-1scale":
        min_max_scaler = preprocessing.MinMaxScaler()
        train_data_std = min_max_scaler.fit_transform(train_data)
        test_data_std = min_max_scaler.fit_transform(test_data)

    elif scale_name == "z-score":
        sc = preprocessing.StandardScaler()
        sc.fit(train_data)
        train_data_std = sc.transform(train_data)
        test_data_std = sc.transform(test_data)

    else:
        print("ERROR:preprocessing")
        # sys.exit()


    return train_data_std,test_data_std


def standardization_rule3(train_data, test_data):

    sc = preprocessing.StandardScaler()
    sc.fit(train_data)
    train_data_std = sc.transform(train_data)
    test_data_std = sc.transform(test_data)

    return train_data_std, test_data_std


def standardization_rule4(train_data, test_data):

    sc = preprocessing.StandardScaler()
    sc.fit(test_data)
    train_data_std = sc.transform(train_data)
    test_data_std = sc.transform(test_data)


    return train_data_std, test_data_std


def standardization(train_data, scale_name):


    if scale_name == "0-1scale":
        min_max_scaler = preprocessing.MinMaxScaler()
        train_data_std = min_max_scaler.fit_transform(train_data)

    elif scale_name == "z-score":
        sc = preprocessing.StandardScaler()
        sc.fit(train_data)
        train_data_std = sc.transform(train_data)

    else:
        print("ERROR:preprocessing")
        # sys.exit()


    return train_data_std




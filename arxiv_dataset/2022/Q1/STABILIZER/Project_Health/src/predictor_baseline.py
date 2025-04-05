import numpy as np
from utils import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
import pdb


def CART(dataset, month, a=12, b=1, c=2):
# def CART(dataset, month, a=5, b=11, c=14):

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    for train, test in df_split(dataset, month):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]
        # max_depth: [1:12], min_samples_leaf: [1:12], min_samples_split: [2:21]

    model = DecisionTreeRegressor(max_depth=a, min_samples_leaf=b, min_samples_split=c)
    model.fit(train_input, train_output)
    test_predict = np.rint(model.predict(test_input))
    test_actual = test_output.values

    mre_list.append(mre_calc(test_predict, test_actual))   ######### for MRE
    sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA
    # print("pre", test_predict, "act", test_actual)
    return mre_list, sa_list


def bell_CART(dataset, month, all_repo, Repo, Directory, goal, a=12, b=1, c=2):
# def CART(dataset, month, a=5, b=11, c=14):

    dataset = normalize(dataset)
    # mre_list = []
    # sa_list = []

    train_input = dataset.iloc[:, :-1]
    train_output = dataset.iloc[:, -1]
    # for train, test in df_split(dataset, month):
    #     train_input = train.iloc[:, :-1]
    #     train_output = train.iloc[:, -1]
    #     test_input = test.iloc[:, :-1]
    #     test_output = test.iloc[:, -1]
        # max_depth: [1:12], min_samples_leaf: [1:12], min_samples_split: [2:21]

    model = DecisionTreeRegressor(max_depth=a, min_samples_leaf=b, min_samples_split=c)
    model.fit(train_input, train_output)

    result_list_mre = {}

    for test_repo in all_repo:
        test_data = data_goal_arrange(test_repo, Directory, goal)
        test_data = normalize(test_data)
        test_input = test_data.iloc[:, :-1]
        test_output = test_data.iloc[:, -1]

        test_predict = np.rint(model.predict(test_input))
        test_actual = test_output.values

        if test_repo not in result_list_mre.keys():
            result_list_mre[test_repo] = []
            # result_list_sa[test_repo] = []

        result_list_mre[test_repo].append(mre_calc(test_predict, test_actual))   ######### for MRE
        # sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA
    # print("pre", test_predict, "act", test_actual)
    return result_list_mre


def KNN(dataset, month):

    n = len(dataset)
    if n < 6:
        n_neighbors = n-1
    else:
        n_neighbors = 5

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    for train, test in df_split(dataset, month):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = neighbors.KNeighborsRegressor(n_neighbors)
    model.fit(train_input, train_output)
    test_predict = np.rint(model.predict(test_input))
    test_actual = test_output.values

    mre_list.append(mre_calc(test_predict, test_actual))   ######### for MRE
    sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA

    return mre_list, sa_list


def SVM(dataset, month):

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    for train, test in df_split(dataset, month):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = svm.SVR(gamma='scale')
    model.fit(train_input, train_output)
    test_predict = np.rint(model.predict(test_input))
    test_actual = test_output.values

    mre_list.append(mre_calc(test_predict, test_actual))   ######### for MRE
    sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA

    return mre_list, sa_list


def RFT(dataset, month, max_depth=3):

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    for train, test in df_split(dataset, month):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = RandomForestRegressor(max_depth)
    model.fit(train_input, train_output)
    test_predict = np.rint(model.predict(test_input))
    test_actual = test_output.values

    mre_list.append(mre_calc(test_predict, test_actual))   ######### for MRE
    sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA

    return mre_list, sa_list


def LNR(dataset, month):

    dataset = normalize(dataset)
    mre_list = []
    sa_list = []
    for train, test in df_split(dataset, month):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = LinearRegression()
    model.fit(train_input, train_output)
    test_predict = np.rint(model.predict(test_input))
    test_actual = test_output.values

    mre_list.append(mre_calc(test_predict, test_actual))   ######### for MRE
    sa_list.append(sa_calc(test_predict, test_actual, train_output))   ######### for SA

    return mre_list, sa_list



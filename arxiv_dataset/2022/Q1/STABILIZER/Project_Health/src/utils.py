from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import preprocessing
import pandas as pd
import numpy as np

import os
import sys
from random import shuffle
import pickle

import json


def df_split(df, month):
    trainData = df.iloc[:-month]
    testData = df.iloc[-month:]
    yield trainData, testData

def df_split_month(df, month):
    trainData = df.iloc[:-month]
    testData = df.iloc[-month:]
    return trainData, testData


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=df.columns, index=df.index)
    lst_col = df.columns[-1]
    df_normalized[lst_col] = df[lst_col]
    return df_normalized


def mre_calc(y_predict, y_actual):
    mre = []
    for predict, actual in zip(y_predict, y_actual):
        if actual == 0:
            if predict == 0:
                mre.append(0)
            elif abs(predict) <= 1:
                mre.append(1)
            else:
                mre.append(round(abs(predict - actual)+1 / (actual+1), 3))
        else:
            mre.append(round(abs(predict - actual) / (actual), 3))
    mMRE = np.median(mre)
    return mMRE


def sa_calc(Y_predict, Y_actual, X_actual):
    Absolute_Error = 0
    for predict, actual in zip(Y_predict, Y_actual):
        Absolute_Error += abs(predict - actual)
    Mean_Absolute_Error = Absolute_Error / (len(Y_predict))
    random_guess = np.mean(X_actual)
    AE_random_guess = 0
    for predict in Y_predict:
        AE_random_guess += abs(predict - random_guess)
    MAE_random_guess = AE_random_guess / (len(Y_predict))
    if MAE_random_guess == 0:
        sa_error = round((1 - (Mean_Absolute_Error+1) / (MAE_random_guess+1)), 3)
    else:
        sa_error = round((1 - Mean_Absolute_Error / MAE_random_guess), 3)
    return sa_error


def confusion_matrix_calc(TN, FP, FN, TP):
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    if TP + FN == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * (Recall * Precision) / (Recall + Precision)
    return [Accuracy, Precision, Recall, F1]


def data_goal_arrange(repo_name, directory, goal):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = get_goal(goal)
    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust


def get_goal(goal):
    if goal == 0:
        last_col = 'monthly_commits'
    elif goal == 1:
        last_col = 'monthly_contributors'
    elif goal == 2:
        last_col = 'monthly_open_PRs'
    elif goal == 3:
        last_col = 'monthly_closed_PRs'
    elif goal == 4:
        last_col = 'monthly_open_issues'
    elif goal == 5:
        last_col = 'monthly_closed_issues'
    elif goal == 6:
        last_col = 'monthly_stargazer'
    else:
        raise Exception("Sorry, wrong goal was passed")
    return last_col


def df_anomaly_label(df):

    last_col_original = df.iloc[:, -1]
    # print(last_col_original)
    df_new = df.copy()
    for i in range(len(df)):
        temp = last_col_original.iloc[:i+1]
        df_new.iloc[i, -1] = np.std(temp)
    sd_column = df_new.iloc[:,-1]
    for i in range(len(sd_column)):
        if i < 12:
            sd_column[i] = 0
    # print(sd_column)
    sd_gain_column = sd_column.copy()
    for i in range(len(sd_gain_column) - 1):
        if sd_column[i] == 0:
            sd_gain_column[i + 1] = 0
        else:
            sd_gain_column[i + 1] = (sd_column[i+1] - sd_column[i]) / sd_column[i]
    # print(sd_gain_column)
    sd_label = sd_gain_column.copy()
    for i in range(len(sd_label)):
        if i < 12:
            sd_label[i] = 0
        elif sd_label[i] < -0.01 or sd_label[i] > 0.1:
            sd_label[i] = 1
        else:
            sd_label[i] = 0
    # pd.set_option("max_rows", None)
    # print(sd_label)
    df_new.iloc[:, -1] = sd_label
    train_part = int(len(sd_label) * 2 / 3)
    first_anomaly = sd_label.idxmax()
    month = train_part if train_part > first_anomaly else first_anomaly
    # print(train_part, first_anomaly, month)
    trainData = df_new.iloc[:month]
    testData = df_new.iloc[month:]
    # pd.set_option("max_columns", 4)
    # print("trainData: ", trainData)
    # print("testData: ", testData)
    yield trainData, testData


def my_confusion_matrix(list_actual, list_predict):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(list_actual)):
        if list_actual[i] == list_predict[i] == 0:
            tn += 1
        if list_actual[i] == list_predict[i] == 1:
            tp += 1
        if list_actual[i] == 0 and list_predict[i] == 1:
            fp += 1
        if list_actual[i] == 1 and list_predict[i] == 0:
            fn += 1
    return tn, fp, fn, tp

def df_norm(dframe, type="normal"):
    """ Normalize a dataframe"""
    col = dframe.columns
    bugs = dframe[dframe.columns[-1]]
    if type == "min_max":
        dframe = (dframe - dframe.min()) / (dframe.max() - dframe.min() + 1e-32)
        dframe[col[-1]] = bugs
        return dframe[col]

    if type == "normal":
        dframe = (dframe - dframe.mean()) / (dframe.std() + + 1e-32)
        dframe[col[-1]] = bugs
        return dframe[col]



if __name__ == '__main__':

    # path = r'../data/data_cleaned/'
    # repo = "abp_monthly.csv"
    #
    # dataset = data_goal_arrange(repo, path, 9)
    # print(len(dataset.columns))

    # for train, test in df_split(dataset, 1):
    #     print(train)
    #     print(test)
    #     for train, test in df_split(train, 1):
    #         print(train)
    #         print(test)

    temp = [[1.1], [2.3], [3.5]]
    print(np.around(temp))
    print(np.rint(temp).astype(np.int))

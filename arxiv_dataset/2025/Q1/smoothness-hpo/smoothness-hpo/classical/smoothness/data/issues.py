from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(filename: str, x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array,
               n_classes: int) -> Tuple[np.array, np.array, np.array, np.array]:
    if n_classes == 2:
        if filename == 'firefox.csv':
            y_train = y_train < 4
            y_test = y_test < 4
        elif filename == 'chromium.csv':
            y_train = y_train < 5
            y_test = y_test < 5
        else:
            y_train = y_train < 6
            y_test = y_test < 6
    elif n_classes == 3:
        y_train = np.where(y_train < 2, 0,
                                np.where(y_train < 6, 1, 2))
        y_test = np.where(
            y_test < 2, 0, np.where(y_test < 6, 1, 2))
    elif n_classes == 5:
        y_train = np.where(y_train < 1, 0, np.where(y_train < 3, 1, np.where(
            y_train < 6, 2, np.where(y_train < 21, 3, 4))))
        y_test = np.where(y_test < 1, 0, np.where(y_test < 3, 1, np.where(
            y_test < 6, 2, np.where(y_test < 21, 3, 4))))
    elif n_classes == 7:
        y_train = np.where(y_train < 1, 0,
                                np.where(y_train < 2, 1, np.where(y_train < 3, 2, np.where(
                                    y_train < 6, 3,
                                    np.where(y_train < 11, 4, np.where(y_train < 21, 5, 6))))))
        y_test = np.where(y_test < 1, 0, np.where(y_test < 2, 1, np.where(y_test < 3, 2, np.where(
            y_test < 6, 3, np.where(y_test < 11, 4, np.where(y_test < 21, 5, 6))))))
    else:
        y_train = np.where(y_train < 1, 0, np.where(y_train < 2, 1, np.where(y_train < 3, 2,
                                                                                            np.where(y_train < 4,
                                                                                                     3, np.where(
                                                                                                    y_train < 6, 4,
                                                                                                    np.where(
                                                                                                        y_train < 8,
                                                                                                        5, np.where(
                                                                                                            y_train < 11,
                                                                                                            6, np.where(
                                                                                                                y_train < 21,
                                                                                                                7,
                                                                                                                8))))))))
        y_test = np.where(y_test < 1, 0, np.where(y_test < 2, 1, np.where(y_test < 3, 2,
                                                                                         np.where(y_test < 4, 3,
                                                                                                  np.where(
                                                                                                      y_test < 6,
                                                                                                      4, np.where(
                                                                                                          y_test < 8,
                                                                                                          5, np.where(
                                                                                                              y_test < 11,
                                                                                                              6,
                                                                                                              np.where(
                                                                                                                  y_test < 21,
                                                                                                                  7,
                                                                                                                  8))))))))

    return x_train, x_test, y_train, y_test


def load_issue_lifetime_prediction_data(filename: str, n_classes: int) -> Tuple[np.array, np.array, np.array, np.array]:
    df = pd.read_csv(f'./data/{filename}.csv')
    df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    _df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
    _df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
    _df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
    _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])
    _df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
    _df['s91'] = df['s9'].apply(lambda x: eval(x)[1])

    if filename == 'firefox':
        _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])

    x = _df.drop('y', axis=1)
    y = _df['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return split_data(filename, x_train, x_test, y_train, y_test, n_classes)

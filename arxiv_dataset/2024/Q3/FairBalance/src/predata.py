import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import copy
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd


def data_preprocess(X):
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(X)
    categorical_columns = categorical_columns_selector(X)

    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('OneHotEncoder', categorical_preprocessor, categorical_columns),
        ('StandardScaler', numerical_preprocessor, numerical_columns)])
    return preprocessor

def fairway(X, y, A):
    preprocessor = data_preprocess(X)
    X_p = preprocessor.fit_transform(X)
    clf1 = LogisticRegression(max_iter=100000)
    clf0 = LogisticRegression(max_iter=100000)
    ambiguous = np.array([True]*len(y))
    for sensitive in A:
        X1=X[X[sensitive] == 1]
        X0=X[X[sensitive] == 0]
        ind1 = X1.index.tolist()
        ind0 = X0.index.tolist()
        y1=y[ind1]
        y0=y[ind0]
        X1_p = preprocessor.transform(X1)
        X0_p = preprocessor.transform(X0)
        clf1.fit(X1_p, y1)
        clf0.fit(X0_p, y0)
        pred1 = clf1.predict(X_p)
        pred0 = clf0.predict(X_p)
        ambiguous = ambiguous & (pred1==pred0)
    new_X = X[ambiguous]
    new_X.index = range(len(new_X))
    new_y = y[X[ambiguous].index]
    return new_X, new_y

def fairsituation(X, y, A):
    clf = LogisticRegression(max_iter=100000)
    preprocessor = data_preprocess(X)
    X_p = preprocessor.fit_transform(X)
    clf.fit(X_p, y)
    pred = clf.predict(X_p)
    ambiguous = np.array([True] * len(y))
    for sensitive in A:
        X_c = copy.deepcopy(X)
        X_c[sensitive] = 1-X_c[sensitive]
        X_cp = preprocessor.transform(X_c)
        pred_c = clf.predict(X_cp)
        ambiguous = ambiguous & (pred == pred_c)
    new_X = X[ambiguous]
    new_X.index = range(len(new_X))
    new_y = y[X[ambiguous].index]
    return new_X, new_y

def fairsmote(X, y, A, cr=0.8, f=0.8):
    def get_ngbr(df, indices):
        rand_sample_idx = random.randint(0, df.shape[0] - 1)
        parent_candidate = df.iloc[rand_sample_idx]
        candidate_2 = df.iloc[indices[rand_sample_idx][1]]
        candidate_3 = df.iloc[indices[rand_sample_idx][2]]
        return parent_candidate, candidate_2, candidate_3

    preprocessor = data_preprocess(X)
    preprocessor.fit(X)
    X_new = copy.deepcopy(X).values.tolist()
    y_new = y.tolist()
    groups_class = {}
    for i in range(len(y)):
        key_class = tuple([X[a][i] for a in A] + [y[i]])
        if key_class not in groups_class:
            groups_class[key_class] = []
        groups_class[key_class].append(i)
    max_count = 0
    for key in groups_class:
        count = len(groups_class[key])
        max_count = max((count, max_count))
    for key in groups_class:
        df = X.loc[groups_class[key]]
        dfp = preprocessor.transform(df)

        knn = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(dfp)
        distances, indices = knn.kneighbors(dfp)
        for i in range(max_count - len(groups_class[key])):
            p, c1, c2 = get_ngbr(df, indices)
            new_point = copy.deepcopy(p)
            for k in X:
                if cr>random.random():
                    if X[k].dtype=='object' or len(np.unique(X[k]))<=2:
                        new_point[k] = random.choice([p[k], c1[k], c2[k]])
                    elif X[k].dtype=='int':
                        new_point[k] = round(p[k]+f*(c1[k] - c2[k]))
                    elif X[k].dtype=='float':
                        new_point[k] = p[k]+f*(c1[k] - c2[k])
            X_new.append(new_point)
            y_new.append(key[-1])
    X_new = pd.DataFrame(X_new)

    columns = list(X.columns)
    for i in range(len(columns)):
        X_new = X_new.rename(columns={i: columns[i]}, errors="raise")

    X_new.index = range(len(X_new))
    return X_new, np.array(y_new)




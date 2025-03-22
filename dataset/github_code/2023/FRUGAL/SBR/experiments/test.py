import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.neighbors import NearestNeighbors
from random import randint, random

import warnings

warnings.filterwarnings("ignore")

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]

train_df = pd.read_csv(f'{root}/data/FARSEC/ambari-train.csv').drop(
    ['id'], axis=1)
test_df = pd.read_csv(f'{root}/data/FARSEC/ambari-test.csv').drop(
    ['id'], axis=1)


# print(train_df)
# print(test_df)

def my_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k - 1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))
    return corpus


def balance(data_train, train_label, m=0, r=0, neighbors=0):
    pos_train = []
    neg_train = []
    for j, i in enumerate(train_label):
        if i == 1:
            pos_train.append(data_train[j])
        else:
            neg_train.append(data_train[j])
    pos_train = np.array(pos_train)
    neg_train = np.array(neg_train)

    if len(pos_train) < len(neg_train):
        pos_train = my_smote(pos_train, m, k=neighbors, r=r)
        if len(neg_train) < m:
            m = len(neg_train)
        neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
    # print(pos_train,neg_train)
    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)
    return data_train1, label_train


def naive_bayes(features, target):
    nb = GaussianNB()
    # nb = MultinomialNB()
    # nb = ComplementNB()
    # nb = BernoulliNB()
    nb.fit(features, target)
    return nb


def result_statistics(predictions):
    # print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    # print("Confusion matrix", confusion_matrix(test_y, predictions))
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print("TN, FP, FN, TP: ", (tn, fp, fn, tp))
    # cm_lr = metrics.confusion_matrix(test_y, predictions)
    # print("Confusion matrix")
    # print(pd.DataFrame(cm_lr))

    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)

    print("pd: ", PD)
    print("pf: ", PF)
    print("prec: ", PREC)
    print("f-measure: ", F_MEASURE)
    print("g-measure: ", G_MEASURE)


train_x = train_df.iloc[:, :-1]
train_y = train_df.iloc[:, -1:]
# train_y = train_dataset['label']

test_x = test_df.iloc[:, :-1]
test_y = test_df.iloc[:, -1:]

# print(train_x)

# lab = [y for x in train_y.values.tolist() for y in x]

# print("Process train_x and train_y with SMOTE balancing")
# train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=256, r=3, neighbors=15)


print("")
print("---------- Default Naive Bayes with StandardScaler----------")

standard_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
print(standard_scaler)

train_x_standard_scaled = standard_scaler.fit_transform(train_x)
test_x_standard_scaled = standard_scaler.transform(test_x)

nb_standardScaler = naive_bayes(train_x_standard_scaled, train_y)
nb_standardScaler_predictions = nb_standardScaler.predict(test_x_standard_scaled)
result_statistics(nb_standardScaler_predictions)

print("")
print("---------- Default Naive Bayes with MinMaxScaler----------")

# Here we scale feature 3 (f3) to a scale between -3 and 3.
# As expected our maximum value (25) is transformed to 3 and
# our minimum value (-1) is transformed to -3.
# All the other values are linearly scaled between these values.
min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-3, 3))
print(min_max_scaler)

train_x_min_max_scaler = min_max_scaler.fit_transform(train_x)
test_x_min_max_scaler = min_max_scaler.transform(test_x)

nb_min_max_scaler = naive_bayes(train_x_min_max_scaler, train_y)
nb_min_max_scaler_predictions = nb_min_max_scaler.predict(test_x_min_max_scaler)
result_statistics(nb_min_max_scaler_predictions)

print("")
print("---------- Default Naive Bayes with MaxAbsScaler----------")

max_abs_scaler = preprocessing.MaxAbsScaler()
print(max_abs_scaler)

train_x_max_abs_scaler = max_abs_scaler.fit_transform(train_x)
test_x_max_abs_scaler = max_abs_scaler.transform(test_x)

nb_max_abs_scaler = naive_bayes(train_x_max_abs_scaler, train_y)
nb_max_abs_scaler_predictions = nb_max_abs_scaler.predict(test_x_max_abs_scaler)
result_statistics(nb_max_abs_scaler_predictions)

print("")
print("---------- Default Naive Bayes with RobustScaler----------")

# By default, the scaler uses the Inter Quartile Range (IQR),
# which is the range between the 1st quartile and the 3rd quartile.
robust_scaler = preprocessing.RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                                           with_scaling=True)
print(robust_scaler)

train_x_robust_scaler = robust_scaler.fit_transform(train_x)
test_x_robust_scaler = robust_scaler.transform(test_x)

nb_robust_scaler = naive_bayes(train_x_robust_scaler, train_y)
nb_robust_scaler_predictions = nb_robust_scaler.predict(test_x_robust_scaler)
result_statistics(nb_robust_scaler_predictions)

print("")
print("---------- Default Naive Bayes with Kernel Centerer----------")

kernel_center = preprocessing.KernelCenterer()
print(kernel_center)

train_x_kernel_center = kernel_center.fit_transform(train_x)
test_x_kernel_center = kernel_center.transform(test_x)

nb_kernel_center = naive_bayes(train_x_kernel_center, train_y)
nb_kernel_center_predictions = nb_kernel_center.predict(test_x_kernel_center)
result_statistics(nb_kernel_center_predictions)

print("")
print("---------- Default Naive Bayes with Quantile Transformation----------")

quantile_transformer = preprocessing.QuantileTransformer(copy=True, n_quantiles=1000, output_distribution='normal',
                                                         random_state=0)
print(quantile_transformer)

train_x_quantile_transformer = quantile_transformer.fit_transform(train_x)
test_x_quantile_transformer = quantile_transformer.transform(test_x)

nb_quantile_transform = naive_bayes(train_x_quantile_transformer, train_y)
nb_quantile_transform_predictions = nb_quantile_transform.predict(test_x_quantile_transformer)
result_statistics(nb_quantile_transform_predictions)

print("")
print("---------- Default Naive Bayes with Power Transformation----------")

power_tranformer = preprocessing.PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
print(power_tranformer)

train_x_power_transformer = power_tranformer.fit_transform(train_x)  # fit to data, then transform it
test_x_power_transformer = power_tranformer.transform(test_x)

nb_power_transform = naive_bayes(train_x_power_transformer, train_y)
nb_power_transform_predictions = nb_power_transform.predict(test_x_power_transformer)
result_statistics(nb_power_transform_predictions)

print("")
print("---------- Default Naive Bayes with Normalization----------")

# The max norm uses the absolute maximum and does for samples what the MaxAbsScaler does for features.
# The l1 norm uses the sum of all the values as and thus gives equal penalty to all parameters, enforcing sparsity.
# The l2 norm uses the square root of the sum of all the squared values. This creates smoothness and rotational invariance.
# Some models, like PCA, assume rotational invariance, and so l2 will perform better.

normalizer = preprocessing.Normalizer(copy=True, norm='l2')
print(normalizer)

train_x_normalizer = normalizer.fit_transform(train_x)
test_x_normalizer = normalizer.transform(test_x)

nb_normalizer = naive_bayes(train_x_normalizer, train_y)
nb_normalizer_predictions = nb_normalizer.predict(test_x_normalizer)
result_statistics(nb_normalizer_predictions)

print("")
print("---------- Default Naive Bayes with Feature Binarization----------")

binarizer = preprocessing.Binarizer(copy=True,
                                    threshold=1.0)  # Feature values below or equal to this are replaced by 0, above it by 1.
print(binarizer)

train_x_binarizer = binarizer.fit_transform(train_x)
test_x_binarizer = binarizer.transform(test_x)

nb_binarizer = naive_bayes(train_x_binarizer, train_y)
nb_binarizer_predictions = nb_binarizer.predict(test_x_binarizer)
result_statistics(nb_binarizer_predictions)

print("")
print("---------- Default Naive Bayes with Polynomial Features----------")

# when degree is set to two and X=x1, x2, the features created will be 1, x1, x2, x1², x1x2 and x2²
# The interaction_only parameter let the function know we only want the interaction features, i.e. 1, x1, x2 and x1x2.
polynomial = preprocessing.PolynomialFeatures(2, interaction_only=False)  # The degree of the polynomial features.
print(polynomial)

train_x_polynomial = polynomial.fit_transform(train_x)
test_x_polynomial = polynomial.transform(test_x)

nb_polynomial = naive_bayes(train_x_polynomial, train_y)
nb_polynomial_predictions = nb_polynomial.predict(test_x_polynomial)
result_statistics(nb_polynomial_predictions)



print("")
print("")
print("test...")

print("training data set")
print(train_df)

print("train_x")
print(type(train_x))

print("train_y")
print(train_y)

print("processed training data set")
process_training = train_x.copy()
sLength = len(process_training.index)
process_training['label'] = train_y.values

process_training_cp = process_training.assign(label=train_y.values)
print(process_training_cp)

# print(process_training)

print(train_df.equals(process_training))
print(train_df.equals(process_training_cp))

# KNN(train_df, test_df, HP)

print("")
print("---------- Default Naive Bayes with StandardScaler----------")

transformer = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
print(standard_scaler)

train_x_copy = train_x.copy()
train_x_transformed = transformer.fit_transform(train_x_copy)
test_x_copy = test_x.copy()
test_x_transformed = transformer.transform(test_x_copy)  # TODO check here


print(type(train_x_copy))
print(train_x_copy)

column_name = list(train_x_copy.columns)
print(column_name)

print(type(train_x_transformed))
train_x_transformed_df = pd.DataFrame(train_x_transformed)
train_x_transformed_df.columns = column_name

print(train_x_transformed_df)

train_df_transformed = train_x_transformed_df.assign(label=train_y.values)
# test_df_transformed = test_x_transformed_df.assign(label=test_y.values)

print(train_df_transformed)
# print(test_df_transformed)

# train_x_standard_scaled = standard_scaler.fit_transform(train_x)
# test_x_standard_scaled = standard_scaler.transform(test_x)
#
# nb_standardScaler = naive_bayes(train_x_standard_scaled, train_y)
# nb_standardScaler_predictions = nb_standardScaler.predict(test_x_standard_scaled)
# result_statistics(nb_standardScaler_predictions)

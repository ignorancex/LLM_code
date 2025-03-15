from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


number_of_iterations = 1
gammas = [0]
current_gamma = 0
for i in range(number_of_iterations):
    current_gamma = (1 + sqrt(1 + 4 * current_gamma * current_gamma)) / 2
    gammas.append(current_gamma)

data = pd.read_csv('super_conduct_train_1000_mcar40.csv')
test_data = pd.read_csv('super_conduct_test_data.csv')
d = str(81)


X_test = test_data[test_data.columns[0:-1]]
Y_test = test_data[[test_data.columns[-1]]]


X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

number_of_test_points = Y_test.shape[0]
original_std = np.nanstd(Y_test)

data_points = data.shape[0]
column_numbers = []
i = 0

nulls = data.isnull().sum(axis=0)
for item in nulls:
    # print(item / data_points)
    if item / data_points > 0.95:
        column_numbers.append(data.columns[i])

    i += 1
X = data[data.columns[0:-1]]

# X = X.drop(column_numbers, axis=1)

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[['critical_temp']]

train_std = sqrt(Y.var()[0])
train_mean = Y.mean()[0]

mask_Y_test = Y.isna()
mask_Y_test = mask_Y_test.to_numpy()
missing_entries = mask_Y_test.sum(axis=0)[0]
mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

cols = X.columns
inds = X.index

cols_y = Y.columns
inds_y = Y.index

X1 = sc.transform(X)
Y1 = sc_y.transform(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns
print(features)

sample_coeff = 1
sampling_number = 30

with_replacement = True

msk_gram = np.dot(mask_X.T, mask_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train_X[[feature_i, feature_j]]
        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram[i][j]:
            print(intersection_num)
            print(msk_gram[i][j])
            print(i, j)
            print("Error")
            exit(1)

        sample_size = intersection_num // sample_coeff
        if sample_size < 5:
            sample_size = intersection_num
            with_replacement = True

        estimation_array = []

        for ind in range(sampling_number):
            current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
            with_replacement = False

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]
            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        confidence_matrix[i][j] = np.std(estimation_array)
        # print(estimation_array)
        # print(i, j, C[i][j], confidence_matrix[i][j])

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        confidence_matrix[i][j] = confidence_matrix[j][i]

print("------------Confidence Matrix---------------")
print(confidence_matrix)
print("---------------------------")
# target confidence:
conf_list = []

cov_msk_train = np.dot(mask_X.T, mask_Y)

for i in range(number_of_features):
    feature_i = features[i]
    current_feature = train_X[[feature_i]].to_numpy()
    current_Y = train_Y.to_numpy()

    columns = np.concatenate((current_feature, current_Y), axis=1)

    columns = pd.DataFrame(columns, columns=[feature_i, 'Y'])
    intersections = columns[columns[[feature_i, "Y"]].notnull().all(axis=1)]
    intersections2 = columns[columns[[feature_i]].notnull().all(axis=1)]

    intersection_num = len(intersections)
    intersection_num2 = len(intersections2)
    if intersection_num != cov_msk_train[i][0]:
        print(intersection_num, intersection_num2, cov_msk_train[i][0])
        exit(1)

    sample_size = intersection_num // sample_coeff
    estimation_array = []

    for ind in range(sampling_number):
        current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

np.savetxt("conf_matrix.csv", confidence_matrix, delimiter=",")
np.savetxt("conf_list.csv", conf_list, delimiter=",")

# confidence_matrix = np.loadtxt('conf_matrix.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list.csv', delimiter=',')

print(confidence_matrix.shape)
print(conf_list)
const = 100
C_min = C - const * confidence_matrix
C_max = C + const * confidence_matrix

y_conf = np.asarray(conf_list)
y_conf = y_conf[:, np.newaxis]
print(y_conf.shape)
print(b.shape)

b_min = b - const * y_conf
b_max = b + const * y_conf

step_size = 0.0001

lam_list = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

print("----------------------")
for lam in lam_list:

    best_C = np.ones(shape=C.shape)
    best_b = np.ones(shape=b.shape)
    best_rmse = 999999999999

    theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)
    ident = np.identity(C.shape[0])

    check_every_iteration = False

    currentC = C.copy()
    currentB = b.copy()

    previous_rmse = 0
    current_rmse = 0
    current_y_C = currentC
    current_y_b = currentB

    previousC = currentC
    previousB = currentB
    stores = []
    iters = []

    for k in range(number_of_iterations):

        # Extrapolation
        current_y_C = currentC + (gammas[k] - 1) / gammas[k + 1] * (currentC - previousC)
        previousC = currentC

        current_y_b = currentB + (gammas[k] - 1) / gammas[k + 1] * (currentB - previousB)
        previousB = currentB

        # Projected Gradient Ascent
        currentC = current_y_C + step_size * np.dot(theta, theta.T)
        currentC = np.clip(currentC, C_min, C_max)

        currentB = current_y_b - 2 * step_size * theta
        currentB = np.clip(currentB, b_min, b_max)

        theta = np.dot(np.linalg.inv(currentC + lam * ident), currentB)

    Y_pred = np.dot(X_test, theta)
    Y_pred = train_std * Y_pred + train_mean
    mse = np.linalg.norm(Y_pred - Y_test) ** 2 / number_of_test_points
    print(lam)
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    print("-----------------------------")


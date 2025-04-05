from py_qt import bootstrap as bs
import matplotlib.pyplot as plt
from py_qt import npr_methods
import numpy as np

from py_qt import nonparam_regression as smooth
from py_qt import plot_fit
import tensorflow as tf
import requests


def f(x):
    return 3*np.cos(x/2) + x**2/5 + 3

xs = np.random.rand(200) * 10
ys = f(xs) + 2*np.random.randn(*xs.shape)

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
# Pull out target variable
y_vals = np.array([x[1] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([x[2:9] for x in birth_data])

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# xs = x_vals_train
# ys = y_vals_train
grid = np.r_[0:10:512j]

k1 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel())
k1.fit()

plt.figure()
# plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
plt.plot(k1(grid), 'k', label='quadratic', linewidth=2)
plt.show()

# grid = np.r_[0:10:512j]
#
# plt.plot(grid, f(grid), 'r--', label='Reference')
# plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
# # plt.legend(loc='best')
# # plt.show()
#
# k0 = smooth.NonParamRegression(xs, ys, method=npr_methods.SpatialAverage())
# k0.fit()
# plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
# plt.legend(loc='best')
#
# k1 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=1))
# k2 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#
# k1.fit()
# k2.fit()
#
# plt.figure()
# plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
# plt.plot(grid, k2(grid), 'k', label='quadratic', linewidth=2)
# plt.plot(grid, k1(grid), 'g', label='linear', linewidth=2)
# plt.plot(grid, f(grid), 'r--', label='Target', linewidth=2)
# plt.legend(loc='best')
#
# yopts = k2(xs)
# res = ys - yopts
# plot_fit.plot_residual_tests(xs, yopts, res, 'Local Quadratic')
#
# yopts = k1(xs)
# res = ys - yopts
# plot_fit.plot_residual_tests(xs, yopts, res, 'Local Linear')
# plt.show()

# def fit(xs, ys):
#     est = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#     est.fit()
#     return est
#
# result = bs.bootstrap(fit, xs, ys, eval_points = grid, CI = (95,99))
#
# plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
# plt.plot(grid, result.y_fit(grid), 'r', label="Fitted curve", linewidth=2)
# plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
# plt.legend(loc=0)
# plt.show()




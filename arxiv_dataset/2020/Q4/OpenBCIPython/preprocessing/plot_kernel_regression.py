"""
========================================================================
Comparison of kernel regression (KR) and support vector regression (SVR)
========================================================================

Toy example of 1D regression using kernel regression (KR) and support vector
regression (SVR). KR provides an efficient way of selecting a kernel's
bandwidth via leave-one-out cross-validation, which is considerably faster
that an explicit grid-search as required by SVR. The main disadvantages are
that it does not support regularization and is not robust to outliers.
"""
from py_qt import bootstrap as bs
import matplotlib.pyplot as plt
from py_qt import npr_methods
import numpy as np

from py_qt import nonparam_regression as smooth
from py_qt import plot_fit
import tensorflow as tf
import requests



import time

import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt

from kernel_regression import KernelRegression

np.random.seed(0)

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
###############################################################################
# Generate sample data
# X = np.sort(5 * np.random.rand(100, 1), axis=0)
# y = np.sin(X).ravel()

X=x_vals_train
y=y_vals_train

###############################################################################
# Add noise to targets
y += 0.5 * (0.5 - np.random.rand(y.size))

###############################################################################
# Fit regression models
svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e-1, 1e0, 1e1, 1e2],
                               "gamma": np.logspace(-2, 2, 10)})
kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
t0 = time.time()
y_svr = svr.fit(X, y).predict(X)
print("SVR complexity and bandwidth selected and model fitted in %.3f s" \
    % (time.time() - t0))
t0 = time.time()
y_kr = kr.fit(X, y).predict(X)
print("KR including bandwith fitted in %.3f s"% (time.time() - t0))

###############################################################################
# Visualize models
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_kr, c='g', label='Kernel Regression')
plt.plot(X, y_svr, c='r', label='SVR')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Kernel regression versus SVR')
plt.legend()

# Visualize learning curves
plt.figure()
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, X, y, train_sizes=np.linspace(0.1, 1, 10),
                   scoring="mean_squared_error", cv=10)
train_sizes_abs, train_scores_kr, test_scores_kr = \
    learning_curve(kr, X, y, train_sizes=np.linspace(0.1, 1, 10),
                   scoring="mean_squared_error", cv=10)
plt.plot(train_sizes, test_scores_svr.mean(1), 'o-', color="r",
         label="SVR")
plt.plot(train_sizes, test_scores_kr.mean(1), 'o-', color="g",
         label="Kernel Regression")
plt.yscale("symlog", linthreshy=1e-7)
plt.ylim(-10, -0.01)
plt.xlabel("Training size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
plt.show()

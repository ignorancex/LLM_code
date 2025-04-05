# mdlcpstat.py
import numpy as np
import scipy.linalg as sl
from scipy import special
from collections import deque
import functools as fts


class batch:
    """
    Differential MDL Change Statistics (batch)

    parameters:
        lossfunc: encoding function.
        d: dimentionality of parameter. For Gaussian distribution, d=2.
        alpha: upper-bound of error probability for 1st and 2nd D-MDL.
        delta: parameter for asymptotic reliability.
        how_to_drop: 'cutpoint' or 'all'.
        preprocess: to encode data with linear regression (True) or not (False). 
        complexity: complexity of encoding function.

    attributes:

    methods:
    decision_function(X): calculate statistics
        input: X ... time series vector
        output: n dimensional vector * 8 ... window size, cutpoint, mdl_0, mdl_1, mdl_2, indice of alarms
    """

    def __init__(self, lossfunc, d, alpha, delta, how_to_drop, preprocess=False, complexity=None):
        self.lossfunc = loss_gaussian if lossfunc is None else lossfunc
        self.complexity = complexity
        self.how_to_drop = how_to_drop
        self.d = d
        self.delta = delta
        self.alpha = alpha
        self.window = deque([])
        self._threshold = 0.5
        self.preprocess = preprocess
        assert how_to_drop in ('cutpoint', 'all')

    def decision_function(self, X):
        detector = online(lossfunc=self.lossfunc, d=self.d, alpha=self.alpha,
                          delta=self.delta, how_to_drop=self.how_to_drop, preprocess=self.preprocess, complexity=self.complexity)

        ret_MDL_0 = []
        ret_window = []
        ret_cut = []
        ret_MDL_1 = []
        ret_MDL_2 = []
        ret_alarm_0 = []
        ret_alarm_1 = []
        ret_alarm_2 = []

        for i, X_i in enumerate(X):
            windowsize, cut, MDL_0, MDL_1, MDL_2, alarm_0, alarm_1, alarm_2 = detector.update(
                X_i)

            ret_window.append(windowsize)
            ret_cut.append(cut)
            ret_MDL_0.append(MDL_0)
            ret_MDL_1.append(MDL_1)
            ret_MDL_2.append(MDL_2)
            ret_alarm_0.append(alarm_0)
            ret_alarm_1.append(alarm_1)
            ret_alarm_2.append(alarm_2)

        ret_alarm_0 = np.array(ret_alarm_0)
        ret_alarm_1 = np.array(ret_alarm_1)
        ret_alarm_2 = np.array(ret_alarm_2)

        if self.preprocess == False:

            ret_alarm_0 = np.where(ret_alarm_0 == 1)[0]
            ret_alarm_1 = np.where(ret_alarm_1 == 1)[0]
            ret_alarm_2 = np.where(ret_alarm_2 == 1)[0]

            return np.array(ret_window), np.array(ret_cut), np.array(ret_MDL_0), np.array(ret_MDL_1), np.array(ret_MDL_2), np.array(ret_alarm_0), np.array(ret_alarm_1), np.array(ret_alarm_2)
        else:

            ret_alarm_0_p = np.where(ret_alarm_0 == 1)[0]
            ret_alarm_0_m = np.where(ret_alarm_0 == -1)[0]
            ret_alarm_1 = np.where(ret_alarm_1 == 1)[0]
            ret_alarm_2 = np.where(ret_alarm_2 == 1)[0]
            return np.array(ret_window), np.array(ret_cut), np.array(ret_MDL_0), np.array(ret_MDL_1), np.array(ret_MDL_2), np.array(ret_alarm_0_p), np.array(ret_alarm_0_m), np.array(ret_alarm_1), np.array(ret_alarm_2)


class online:
    """
    Differential MDL Change Statistics (online)

    parameters:
        lossfunc: encoding function.
        d: dimentionality of parameter. For Gaussian distribution, d=2.
        alpha: upper-bound of error probability for 1st and 2nd D-MDL.
        delta: parameter for asymptotic reliability.
        how_to_drop: 'cutpoint' or 'all'.
        preprocess: to encode data with linear regression (True) or not (False). 
        complexity: complexity of encoding function.

    attributes:

    methods:
    update(X): calculate statistics
        input: X ... data at a time point
        output: statistics ... window size, cutpoint, mdl_0, mdl_1, mdl_2, indice of alarms
    """

    def __init__(self, lossfunc, d, alpha, delta, how_to_drop, preprocess=False, complexity=None):
        self.lossfunc = loss_gaussian if lossfunc is None else lossfunc
        self.complexity = complexity
        self.how_to_drop = how_to_drop
        self.d = d
        self.delta = delta
        self.alpha = alpha
        self.window = deque([])
        self.preprocess = preprocess
        assert how_to_drop in ('cutpoint', 'all')

    def update(self, X):
        self.window.append(X)
        if len(self.window) == 1:
            return 1, -1, 0, np.nan, np.nan, 0, 0, 0
        list_window = list(self.window)
        MDL_0, MDL_1, MDL_2, alarm_0, alarm_1, alarm_2 = _mdlcstat_adwin(
            self.lossfunc, list_window, delta=self.delta, d=self.d, alpha=self.alpha, preprocess=self.preprocess, complexity=self.complexity)
        i = len(MDL_0) - 1
        nn = len(list_window)
        #print(MDL_0)
        #print(calculate_threshold(delta=self.delta, d=self.d, nn=nn - 13, alpha=self.alpha, n=nn, order=0, complexity=self.complexity))

        if nn>=10:
            #print(calculate_threshold(delta=self.delta, d=self.d, nn=nn - 5, alpha=self.alpha, n=nn, order=0, complexity=self.complexity))
            while MDL_0[i] <= calculate_threshold(delta=self.delta, d=self.d, nn=nn - 9, alpha=self.alpha, n=nn, order=0, complexity=self.complexity):
                i = i - 1
                if i < 0:
                    break
        else:
            i=-1
        cut = i
        if cut != -1:
            if self.how_to_drop == 'cutpoint':
                cut = np.argmax(MDL_0)
                for j in range(0, cut + 1):
                    self.window.popleft()
                return len(self.window), max(MDL_0), MDL_1, MDL_2, alarm_0, alarm_1, alarm_2
            if self.how_to_drop == 'all':
                self.window = deque([])
                return 0, cut, max(MDL_0), MDL_1, MDL_2, alarm_0, alarm_1, alarm_2
        else:
            return len(self.window), -1, max(MDL_0), MDL_1, MDL_2, alarm_0, alarm_1, alarm_2


def calculate_threshold(delta, d, nn, alpha, n, order=0, complexity=None):
    """
    calculate threshold for an order of D-MDL.

    parameters:
        delta: the hyperparameter for asymptotic reliability
        d: dimention of parameter
        nn: the number of possible cutpoints
        alpha: upper-bound of error probability for 1st and 2nd D-MDL
        n: the number of data within the window
        order: 0th, 1st, or 2nd
        complexity(optional): the function for calculating stochastic complexity. If None,
            use asymptotic expansion instead.

    returns:
        threshold
    """
    if order == 0:
        threshold = np.log(1 / delta) + (d / 2 + 1 + delta) * np.log(n) + np.log(nn)
    elif order == 1:
        threshold = d * np.log(n / 2) - np.log(alpha)
    else:
        threshold = 2 * (d * np.log(n / 2) - np.log(alpha))
    #print(order)
    #print(threshold)

    '''
    if order == 0:  # 0th order MDL change statistics. This fulfills the asymptotic reliability.
        if complexity == None:
            threshold = np.log(1 / delta) + (d / 2 + 2 + delta) * np.log(nn)
        else:
            threshold = np.log(1 / delta) + (2 + delta) * \
                np.log(nn) + complexity(n)
    else:
        if complexity == None:
            if order == 1:
                threshold = d * np.log(n / 2) - np.log(alpha)
            else:
                threshold = 2 * (d * np.log(n / 2) - np.log(alpha))

        else:
            if order == 1:
                threshold = 2 * complexity(n / 2) - np.log(alpha)
            else:
                threshold = 2 * (2 * complexity(n / 2) - np.log(alpha))
    '''
    return threshold


def calculate_residual(X, grad=False):
    """
    calculate residual error of data with linear regression 

    parameters:
        X: data sequence
        grad: return gradient of the result of linear regression or not.

    returns:
        residual error, gradient(optional)
    """

    if X.size == 1:
        if grad == False:
            return np.zeros((1, 1))
        else:
            return np.zeros((1, 1)), 100
    elif X.size == 2:
        if grad == False:
            return np.zeros((2, 1))
        else:
            return np.zeros((2, 1)), X[1] - X[0]
    else:
        n = X.shape[0]
        x = np.arange(n)
        y = X[:, 0]
        #a = ((x - np.mean(x)).dot(y - np.mean(y)) / n) / np.var(x)
        # print('a')
        # print(a)
        a = (x.dot(y) / n - np.mean(x) * np.mean(y)) / np.var(x)
        # ((x - np.mean(x)).dot(y - np.mean(y)) / n) / np.var(x)
        # print('a_hat')
        # print(a_hat)

        b = np.mean(y) - a * np.mean(x)
        y = y.reshape((n, 1))
        x = x.reshape((n, 1))
        res = y - (a[0, 0] * x + b[0, 0])
        res = res.reshape((n, 1))
        if grad == False:
            return res
        else:
            return res, a

def _mdlcstat_adwin(lossfunc, X, d, alpha, delta, preprocess=False, complexity=None):
    """
    Calculate each order MDL change statistics for hierarchical algorithm.

    parameters:
        lossfunc: encoding function.
        X: data within the window
        d: dimentionality of parameter. For Gaussian distribution, d=2.
        alpha: upper-bound of error probability for 1st and 2nd D-MDL.
        delta: parameter for asymptotic reliability.
        preprocess: to encode data with linear regression (True) or not (False). 
        complexity: complexity of encoding function.

    returns:
        Scores for 0th D-MDL, 1st D-DML, 2nd D-MDL, indice of alarms for 0th D-MDL, 1st D-MDL, 2nd D-MDL
    """

    least_datapoints=5

    Xmat = np.matrix(X)
    n, m = Xmat.shape
    if n == 1:
        Xmat = Xmat.T
        n, m = Xmat.shape
    else:
        pass

    if preprocess == False:  # without linear regression

        # calculate sum for computational efficiency
        sum_x = np.zeros((n, m))
        sum_x[0] = Xmat[0]
        for i in range(0, n - 1):
            sum_x[i + 1] = sum_x[i] + Xmat[i + 1]

        # calculate squared sum for computational efficiency
        sum_xxT = np.zeros((n, m, m))
        sum_xxT[0] = Xmat[0].T.dot(Xmat[0])
        for i in range(0, n - 1):
            sum_xxT[i + 1] = sum_xxT[i] + Xmat[i + 1].T.dot(Xmat[i + 1])
        MDL_0 = np.zeros(n - 1)

        MDL_1 = -1e100
        MDL_2 = -1e100
        alarm_0 = 0
        alarm_1 = 0
        alarm_2 = 0

        #print(calculate_threshold(delta=delta, d=d, nn=n - 5, alpha=alpha, n=n, order=0, complexity=complexity))

        # calculate 0th MDL change statistics at each time point
        for cut in range(least_datapoints, n-(least_datapoints-1)):
            L_total = lossfunc(Xmat, sum_x[n - 1], sum_xxT[n - 1])
            L_1 = lossfunc(Xmat[:cut], sum_x[cut - 1], sum_xxT[cut - 1])
            L_2 = lossfunc(Xmat[cut:], sum_x[n - 1] -
                           sum_x[cut - 1], sum_xxT[n - 1] - sum_xxT[cut - 1])
            if L_total - (L_1 + L_2) > calculate_threshold(delta=delta, d=d, nn=n - (least_datapoints*2-1), alpha=alpha, n=n, order=0, complexity=complexity):
                alarm_0 = 1
            MDL_0[cut - 1] = L_total - (L_1 + L_2)

        # calculate 1st MDL change statistics at each time point
        if n < 2*least_datapoints+1:
            MDL_1 = np.nan
        else:
            for cut in range(least_datapoints, n - least_datapoints):
                L_first = lossfunc(Xmat[:cut], sum_x[cut - 1], sum_xxT[cut - 1]) + \
                    lossfunc(Xmat[cut:], sum_x[n - 1] - sum_x[cut - 1],
                             sum_xxT[n - 1] - sum_xxT[cut - 1])
                L_second = lossfunc(Xmat[:cut + 1], sum_x[cut], sum_xxT[cut]) + \
                    lossfunc(Xmat[cut + 1:], sum_x[n - 1] - sum_x[cut],
                             sum_xxT[n - 1] - sum_xxT[cut])
                stat_1 = L_first - L_second
                if stat_1 > calculate_threshold(delta=delta, d=d, nn=n - least_datapoints*2, alpha=alpha, n=n, order=1, complexity=complexity):
                    alarm_1 = 1
                if stat_1 > MDL_1:
                    MDL_1 = stat_1

        # calculate 2nd MDL change statistics at each time point
        if n < 2*least_datapoints+2:
            MDL_2 = np.nan
        else:
            for cut in range(least_datapoints+1, n - least_datapoints):
                # algebric view
                L_t = lossfunc(Xmat[:cut], sum_x[cut - 1], sum_xxT[cut - 1]) + \
                    lossfunc(Xmat[cut:], sum_x[n - 1] - sum_x[cut - 1],
                             sum_xxT[n - 1] - sum_xxT[cut - 1])
                L_tp = lossfunc(Xmat[:cut + 1], sum_x[cut], sum_xxT[cut]) + \
                    lossfunc(Xmat[cut + 1:], sum_x[n - 1] - sum_x[cut],
                             sum_xxT[n - 1] - sum_xxT[cut])
                L_tm = lossfunc(Xmat[:cut - 1], sum_x[cut - 2], sum_xxT[cut - 2]) + \
                    lossfunc(Xmat[cut - 1:], sum_x[n - 1] - sum_x[cut - 2],
                             sum_xxT[n - 1] - sum_xxT[cut - 2])

                stat_2 = 2 * L_t - (L_tp + L_tm)
                if stat_2 > calculate_threshold(delta=delta, d=d, nn=n - (least_datapoints*2+1), alpha=alpha, n=n, order=2, complexity=complexity):
                    alarm_2 = 1
                if stat_2 > MDL_2:
                    MDL_2 = stat_2

        return MDL_0, MDL_1, MDL_2, alarm_0, alarm_1, alarm_2

    else:  # with linear regression

        MDL_0 = np.zeros(n - 1)

        MDL_1 = -1e100
        MDL_2 = -1e100
        alarm_0 = 0
        alarm_1 = 0
        alarm_2 = 0

        # calculate 0th MDL change statistics at each time point
        for cut in range(1, n):
            # calculate residual errors
            res_total = calculate_residual(Xmat)
            res_1, grad_1 = calculate_residual(Xmat[:cut], grad=True)
            res_2, grad_2 = calculate_residual(Xmat[cut:], grad=True)

            # calculate 0th MDL change statistics of residual errors
            L_total = lossfunc(res_total, np.sum(
                res_total).reshape((1, 1)), np.var(res_total))
            L_1 = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1))
            L_2 = lossfunc(res_2, np.sum(res_2).reshape((1, 1)), np.var(res_2))
            if L_total - (L_1 + L_2) > calculate_threshold(delta=delta, d=d, nn=n - 1, alpha=alpha, n=n, order=0, complexity=complexity):
                if grad_1 < grad_2:
                    alarm_0 = 1  # increased gradient
                else:
                    alarm_0 = -1  # decreased gradient
            MDL_0[cut - 1] = L_total - (L_1 + L_2)

        # calculate 1st MDL change statistics at each time point
        if n < 3:
            MDL_1 = np.nan
        else:
            for cut in range(1, n - 1):
                # calculate residual errors and 1st MDL change statistics at
                # time point t
                res_1 = calculate_residual(Xmat[:cut])
                res_2 = calculate_residual(Xmat[cut:])
                L_first = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1)) + \
                    lossfunc(res_2, np.sum(res_2).reshape(
                        (1, 1)), np.var(res_2))

                # calculate residual errors and 1st MDL change statistics at
                # time point t+1
                res_1 = calculate_residual(Xmat[:cut + 1])
                res_2 = calculate_residual(Xmat[cut + 1:])
                L_second = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1)) + \
                    lossfunc(res_2, np.sum(res_2).reshape(
                        (1, 1)), np.var(res_2))

                # calculate 1st MDL change statistics
                stat_1 = L_first - L_second
                if stat_1 > calculate_threshold(delta=delta, d=d, nn=n - 2, alpha=alpha, n=n, order=1, complexity=complexity):
                    alarm_1 = 1
                if stat_1 > MDL_1:
                    MDL_1 = stat_1

        # calculate 2nd MDL change statistics at each time point
        if n < 4:
            MDL_2 = np.nan
        else:
            for cut in range(2, n - 1):
                 # calculate residual errors and 1st MDL change statistics at
                 # time point t
                res_1 = calculate_residual(Xmat[:cut])
                res_2 = calculate_residual(Xmat[cut:])
                L_t = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1)) + \
                    lossfunc(res_2, np.sum(res_2).reshape(
                        (1, 1)), np.var(res_2))

                # calculate residual errors and 1st MDL change statistics at
                # time point t+1
                res_1 = calculate_residual(Xmat[:cut + 1])
                res_2 = calculate_residual(Xmat[cut + 1:])
                L_tp = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1)) + \
                    lossfunc(res_2, np.sum(res_2).reshape(
                        (1, 1)), np.var(res_2))

                # calculate residual errors and 1st MDL change statistics at
                # time point t-1
                res_1 = calculate_residual(Xmat[:cut - 1])
                res_2 = calculate_residual(Xmat[cut - 1:])
                L_tm = lossfunc(res_1, np.sum(res_1).reshape((1, 1)), np.var(res_1)) + \
                    lossfunc(res_2, np.sum(res_2).reshape(
                        (1, 1)), np.var(res_2))

                # calculate 2nd MDL change statistics
                stat_2 = 2 * L_t - (L_tp + L_tm)

                if stat_2 > calculate_threshold(delta=delta, d=d, nn=n - 3, alpha=alpha, n=n, order=2, complexity=complexity):
                    alarm_2 = 1
                if stat_2 > MDL_2:
                    MDL_2 = stat_2

        return MDL_0, MDL_1, MDL_2, alarm_0, alarm_1, alarm_2


@fts.lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)


def nml_regression(X, sum_x, sum_xxT, R=1, var_min=1e-12):
    Xmat = np.matrix(X)

    n = Xmat.shape[0]
    if n <= 0:
        return np.nan

    W = np.ones((2, n))
    W[1, :] = np.arange(1, n + 1)

    beta = sl.pinv(W.dot(W.T)).dot(W).dot(X)
    Xc = Xmat - W.T.dot(beta)
    var = float(Xc.T * Xc / n)

    eps=1e-12
    var=max(var, eps)

    return n * np.log(var)/2 + np.log(R / var_min) - special.gammaln(n / 2 - 1) + n * np.log(n * np.pi) / 2


def complexity_regression(h, R=1, var_min=1e-12):

    return h / 2 * np.log(h / (2 * np.e)) + np.log(R / var_min) + special.gammaln(h / 2 - 1)


def lnml_gaussian(X, sum_x, sum_xxT, sigma_given=1):
    """
    Calculate LNML code length of Gaussian distribution. See the paper below:
    Miyaguchi, Kohei. "Normalized Maximum Likelihood with Luckiness for Multivariate Normal Distributions." 
    arXiv preprint arXiv:1708.01861 (2017).

    parameters: 
        X: data sequence
        sum_x: mean sequence
        sum_xxT: variance sequence
        sigma_given: hyperparameter for prior distribution

    returns:
        LNML code length
    """
    multigammaln = multigamma_ln
    n, m = X.shape
    if n <= 0:
        return np.nan
    nu = m  # given
    sigma = sigma_given  # given
    mu = sum_x / (n + nu)
    S = (sum_xxT + nu * sigma ** 2 * np.matrix(np.identity(m))) / \
        (n + nu) - mu.T.dot(mu)
    detS = sl.det(S)
    # log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
    #    (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)+ 0.5*np.log(n)
    log_lnml = m / 2 * ((nu + 1) * np.log(nu) - n * np.log(np.pi) - (n + nu + 1) * np.log(n + nu)) + multigammaln(
        (n + nu) / 2, m) - multigammaln(nu / 2, m) + m * nu * np.log(sigma) - (n + nu) / 2 * np.log(detS)

    return -1 * log_lnml


def complexity_lnml_gaussian(h, m, sigma_given=1):
    """
    Calculate stochastic complexity of LNML of Gaussian distribution. See the paper below:
    Miyaguchi, Kohei. "Normalized Maximum Likelihood with Luckiness for Multivariate Normal Distributions." 
    arXiv preprint arXiv:1708.01861 (2017).

    parameters: 
        h: half window size
        m: dimension of data
        sigma_given: hyperparameter for prior distribution

    returns:
        stochastics complexity
    """
    multigammaln = multigamma_ln
    n = h
    nu = m  # given
    sigma = sigma_given  # given
    # log_C = -m * nu * np.log(sigma) + multigammaln(nu / 2, m) - multigammaln((nu + n) / 2, m) + 0.5 * m * (n + nu + 1) * np.log(
    # nu + n) - 0.5 * m * (n + nu) * np.log(2) - 0.5 * m * (n + nu) - 0.5 * m
    # * nu * np.log(np.pi) - 0.5 * m * (nu + 1) * np.log(nu) + 0.5*np.log(n)
    log_C = -m * nu * np.log(sigma) + multigammaln(nu / 2, m) - multigammaln((nu + n) / 2, m) + 0.5 * m * (n + nu + 1) * np.log(
        nu + n) - 0.5 * m * (n + nu) * np.log(2) - 0.5 * m * (n + nu) - 0.5 * m * nu * np.log(np.pi) - 0.5 * m * (nu + 1) * np.log(nu)

    return log_C


def app_nml_poisson(X, sum_x, sum_xxT, lmd_max=100):
    """
    Calculate NML code length of Poisson distribution. See the paper below:
    yamanishi, Kenji, and Kohei Miyaguchi. "Detecting gradual changes from data stream using MDL-change statistics." 
    2016 IEEE International Conference on Big Data (Big Data). IEEE, 2016.

    parameters: 
        X: data sequence
        sum_x: mean sequence
        sum_xxT: variance sequence
        lmd_max: the maximum value of lambda

    returns:
        NML code length
    """
    n = len(X)
    lmd_hat = sum_x / n
    if lmd_hat == 0:
        neg_log = np.sum(special.gammaln(X + 1))
    else:
        neg_log = -n * lmd_hat * np.log(lmd_hat) + \
            n * lmd_hat + np.sum(special.gammaln(X + 1))
    cpl = app_complexity_poisson(n, lmd_max)
    return neg_log + cpl


def app_complexity_poisson(h, lmd_max=100):
    """
    Calculate stochastic complexity of Poisson distribution. See the paper below:
    yamanishi, Kenji, and Kohei Miyaguchi. "Detecting gradual changes from data stream using MDL-change statistics." 
    2016 IEEE International Conference on Big Data (Big Data). IEEE, 2016.

    parameters: 
        h: half window size
        lmd_max: the maximum value of lambda

    returns:
        stochastics complexity
    """
    return 0.5 * np.log(h / (2 * np.pi)) + (1 + lmd_max / 2) * np.log(2) + log_star(lmd_max)


def log_star(k):
    ret = np.log(2.865)
    x = k
    while np.log(x) > 0:
        ret += np.log(x)
        x = np.log(x)
    return ret
##############################################################################

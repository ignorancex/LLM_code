import numpy as np
from l2_regression import *
from scipy.optimize import minimize

########################################################
# glse
########################################################
# optimization for glse on the full dataset
def glse_opt(x, *args):
    panel = args[0]
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = x[0:d-1]
    rho = x[d-1:]
    q = len(rho)

    obj = 0
    for i in range(N):
        matrix = panel[i]
        X = matrix[:, 0:-1]
        Y = matrix[:, -1]
        for t in range(T):
            val = Y[t] - np.dot(X[t], beta)
            for tt in range(min(t, q)):
                val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
            obj += val * val
    return obj

####################################
# IRLS for glse on the full dataset

def IRLS_glse(panel, q):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.random.rand(d-1)
    rho = np.random.rand(q)
    value = glse_obj(panel, beta, rho)

    # minimize for beta
    def glse_opt_beta(x, *args):
        beta = x
        rho = args[0]

        obj = 0
        for i in range(N):
            matrix = panel[i]
            X = matrix[:, 0:-1]
            Y = matrix[:, -1]
            for t in range(T):
                val = Y[t] - np.dot(X[t], beta)
                for tt in range(min(t, q)):
                    val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
                obj += val * val
        return obj

    # minimize for rho
    def glse_opt_rho(x, *args):
        beta = args[0]
        rho = x

        obj = 0
        for i in range(N):
            matrix = panel[i]
            X = matrix[:, 0:-1]
            Y = matrix[:, -1]
            for t in range(T):
                val = Y[t] - np.dot(X[t], beta)
                for tt in range(min(t, q)):
                    val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
                obj += val * val
        return obj

    # iteratively optimize over both beta and rho
    flag = 0
    while flag == 0:
        x0 = beta
        res = minimize(glse_opt_beta, x0, args = rho, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        beta = res.x
        x0 = rho
        res = minimize(glse_opt_rho, x0, args = beta, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        rho = res.x
        temp_value = glse_obj(panel, beta, rho)
        if temp_value >= value - 1e-8:
            flag = 1
        value = temp_value

    obj = value
    return beta, rho, obj

#####################################
# optimization for glse on the coreset
def glse_coreset_opt(x,*args):
    panel = args[0]
    coreset = args[1]
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = x[0:d-1]
    rho = x[d-1:]
    q = len(rho)
    obj = 0
    for c in range(len(coreset)):
        t = coreset[c][0] % T
        i = int((coreset[c][0]-t)/T)
        matrix = panel[i]
        X = matrix[:,0:-1]
        Y = matrix[:,-1]
        val = Y[t] - np.dot(X[t],beta)
        for tt in range(min(t, q)):
            val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
        obj += coreset[c][1] * val * val
    return obj

####################################
# IRLS for glse on the coreset

def IRLS_glse_coreset(panel, coreset, q):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.random.rand(d - 1)
    rho = np.random.rand(q)
    value = glse_coreset_obj(panel, beta, rho, coreset)

    # minimize for beta
    def glse_coreset_opt_beta(x, *args):
        beta = x
        rho = args[0]

        obj = 0
        for c in range(len(coreset)):
            t = coreset[c][0] % T
            i = int((coreset[c][0] - t) / T)
            matrix = panel[i]
            X = matrix[:, 0:-1]
            Y = matrix[:, -1]
            val = Y[t] - np.dot(X[t], beta)
            for tt in range(min(t, q)):
                val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
            obj += coreset[c][1] * val * val
        return obj

    # minimize for rho
    def glse_coreset_opt_rho(x, *args):
        beta = args[0]
        rho = x

        obj = 0
        for c in range(len(coreset)):
            t = coreset[c][0] % T
            i = int((coreset[c][0] - t) / T)
            matrix = panel[i]
            X = matrix[:, 0:-1]
            Y = matrix[:, -1]
            val = Y[t] - np.dot(X[t], beta)
            for tt in range(min(t, q)):
                val -= rho[tt] * (Y[t - tt - 1] - np.dot(X[t - tt - 1], beta))
            obj += coreset[c][1] * val * val
        return obj

    # iteratively optimize over both beta and rho
    flag = 0
    while flag == 0:
        x0 = beta
        res = minimize(glse_coreset_opt_beta, x0, args = rho, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        beta = res.x
        x0 = rho
        res = minimize(glse_coreset_opt_rho, x0, args = beta, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        rho = res.x
        temp_value = glse_coreset_obj(panel, beta, rho, coreset)
        if temp_value >= value - 1e-8:
            flag = 1
        value = temp_value

    obj = glse_obj(panel,beta,rho)
    return beta, rho, obj

#####################################
# glsek
#####################################
# optimization for glsek on the full dataset
def glsek_opt(x, *args):
    panel = args[0]
    q = args[1][0]
    k = args[1][1]
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.array([np.random.rand(d-1) for j in range(k)])
    rho = np.array([np.random.rand(q) for j in range(k)])
    for i in range(k):
        beta[i] = x[i*(d-1):(i+1)*(d-1)]
    index = k * (d-1)
    for i in range(k):
        rho[i] = x[i*q+index:(i+1)*q+index]

    obj = 0
    for i in range(N):
        matrix = np.array([panel[i]])
        val = [0 for i in range(k)]
        for l in range(k):
            val[l] = glse_obj(matrix,beta[l],rho[l])
        obj += min(val)
    return obj

####################################
# IRLS for glsek on the full dataset

def IRLS_glsek(panel, q, k):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.array([[0.0 for i in range(d - 1)] for j in range(k)])
    rho = np.array([[0.0 for i in range(q)] for j in range(k)])
    value = glsek_obj(panel, beta, rho)

    # minimize for beta
    def glsek_opt_beta(x, *args):
        beta = x.reshape(-1,d-1)
        rho = args[0]

        obj = 0
        for i in range(N):
            matrix = np.array([panel[i]])
            val = [0 for i in range(k)]
            for l in range(k):
                val[l] = glse_obj(matrix, beta[l], rho[l])
            obj += min(val)
        return obj

    # minimize for rho
    def glsek_opt_rho(x, *args):
        beta = args[0]
        rho = x.reshape(-1,q)

        obj = 0
        for i in range(N):
            matrix = np.array([panel[i]])
            val = [0 for i in range(k)]
            for l in range(k):
                val[l] = glse_obj(matrix, beta[l], rho[l])
            obj += min(val)
        return obj

    # iteratively optimize over both beta and rho
    flag = 0
    while flag == 0:
        x0 = beta.flatten()
        res = minimize(glsek_opt_beta, x0, args = rho, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        beta = np.array(res.x)
        beta = beta.reshape(-1,d-1)
        x0 = rho.flatten()
        res = minimize(glsek_opt_rho, x0, args = beta, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        rho = np.array(res.x)
        rho = rho.reshape(-1,q)
        temp_value = glsek_obj(panel, beta, rho)
        if temp_value >= value - 1e-8:
            flag = 1
        value = temp_value

    obj = value
    return beta, rho, obj

#####################################
# optimization for glsek on the coreset
def glsek_coreset_opt(x, *args):
    panel = args[0]
    coreset = args[1]
    q = args[2][0]
    k = args[2][1]
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.array([np.random.rand(d-1) for j in range(k)])
    rho = np.array([np.random.rand(q) for j in range(k)])
    for i in range(k):
        beta[i] = x[i * (d - 1):(i + 1) * (d - 1)]
    index = k * (d - 1)
    for i in range(k):
        rho[i] = x[i * q + index:(i + 1) * q + index]

    obj = 0
    for i in range(len(coreset)):
        id = coreset[i]
        matrix = np.array([panel[id[0]]])
        time_coreset = id[1:]
        val = [0 for i in range(k)]
        for l in range(k):
            val[l] = glse_coreset_obj(matrix, beta[l], rho[l],time_coreset)
        obj += min(val)
    return obj

####################################
# IRLS for glsek on the coreset

def IRLS_glsek_coreset(panel, coreset, q, k):
    N = panel.shape[0]
    T = panel.shape[1]
    d = panel.shape[2]
    beta = np.array([[0.0 for i in range(d - 1)] for j in range(k)])
    rho = np.array([[0.0 for i in range(q)] for j in range(k)])
    value = glsek_coreset_obj(panel, beta, rho, coreset)

    # minimize for beta
    def glsek_coreset_opt_beta(x, *args):
        beta = x.reshape(-1,d-1)
        rho = args[0]

        obj = 0
        for i in range(len(coreset)):
            id = coreset[i]
            matrix = np.array([panel[id[0]]])
            time_coreset = id[1:]
            val = [0 for i in range(k)]
            for l in range(k):
                val[l] = glse_coreset_obj(matrix, beta[l], rho[l], time_coreset)
            obj += min(val)
        return obj

    # minimize for rho
    def glsek_coreset_opt_rho(x, *args):
        beta = args[0]
        rho = x.reshape(-1,q)

        obj = 0
        for i in range(len(coreset)):
            id = coreset[i]
            matrix = np.array([panel[id[0]]])
            time_coreset = id[1:]
            val = [0 for i in range(k)]
            for l in range(k):
                val[l] = glse_coreset_obj(matrix, beta[l], rho[l], time_coreset)
            obj += min(val)
        return obj

    # iteratively optimize over both beta and rho
    flag = 0
    while flag == 0:
        x0 = beta.flatten()
        res = minimize(glsek_coreset_opt_beta, x0, args = rho, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        beta = np.array(res.x)
        beta = beta.reshape(-1,d-1)
        x0 = rho.flatten()
        res = minimize(glsek_coreset_opt_rho, x0, args = beta, method='SLSQP', options={'ftol': 1e-8, 'eps' : 1e-8, 'disp': False})
        rho = np.array(res.x)
        rho = rho.reshape(-1,q)
        temp_value = glsek_coreset_obj(panel, beta, rho, coreset)
        if temp_value >= value - 1e-8:
            flag = 1
        value = temp_value

    obj = glsek_obj(panel,beta,rho)
    return beta, rho, obj

#####################################

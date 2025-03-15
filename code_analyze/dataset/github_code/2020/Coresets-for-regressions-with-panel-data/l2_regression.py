import numpy as np

#####################################
# the objective function of glse on the full dataset
def glse_obj(panel,beta,rho):
    N = panel.shape[0]
    T = panel.shape[1]
    q = len(rho)

    obj = 0
    for i in range(N):
        matrix = panel[i]
        X = matrix[:,0:-1]
        Y = matrix[:,-1]
        for t in range(T):
            val = Y[t] - np.dot(X[t],beta)
            for tt in range(min(t,q)):
                val -= rho[tt] * (Y[t-tt-1] - np.dot(X[t-tt-1],beta))
            obj += val * val
    return obj

#####################################
# the objective function of glse on the coreset
def glse_coreset_obj(panel,beta,rho,coreset):
    T = panel.shape[1]
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

#####################################
# the objective function of glsek on the full dataset
def glsek_obj(panel,beta,rho):
    N = panel.shape[0]
    k = len(rho)

    obj = 0
    for i in range(N):
        matrix = np.array([panel[i]])
        val = [0 for i in range(k)]
        for l in range(k):
            val[l] = glse_obj(matrix,beta[l],rho[l])
        obj += min(val)
    return obj

#####################################
# the objective function of glsek on the coreset
def glsek_coreset_obj(panel,beta,rho,coreset):
    k = len(rho)

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


#########################################

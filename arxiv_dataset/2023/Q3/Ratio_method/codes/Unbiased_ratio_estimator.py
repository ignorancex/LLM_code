import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

def unbiased_R_estimator(d1,d2,cov1,cov2):
    deltaR=0.001 
    R = np.arange(1,3,deltaR) # try various R
    NR = len(R)
    
    def Matrix3(M1,M2,M3): # Multiplication of three matrices
        return np.dot(np.dot(M1,M2),M3)

    def prob_R(d1,d2,cov1,cov2,R):
        from numpy.linalg import inv
        n = d1.shape[0]
        a1 = np.identity(n) # the mapping matrix A1, A2 is indentity
        a2 = a1 * R         # Eq.19 in paper
        q1 = Matrix3(a1.T, inv(cov1), a1)  
        q2 = Matrix3(a2.T, inv(cov2), a2)
        q = q1 + q2  # defination is below Eq.5
        d1 = d1.reshape(1,n)
        d2 = d2.reshape(1,n)
        t1 = Matrix3(a1.T, inv(cov1), d1.T)
        t2 = Matrix3(a2.T, inv(cov2), d2.T)
        t = t1 + t2  # defination is below Eq.5
        return 0.5 * Matrix3(t.T, inv(q), t), np.sqrt(np.linalg.det(q)) # two terms in Eq.8

    def pdf(d1,d2,cov1,cov2,R):
        Exp_index=[]; DetQ=[]
        for rr in range(NR):
            exp_index, detQ = prob_R(d1=d1,d2=d2,cov1=cov1,cov2=cov2,R=R[rr])
            Exp_index.append(exp_index)
            DetQ.append(detQ)
        Exp_index = np.array(Exp_index).reshape(NR)
        index_max = max(Exp_index)

        # To avoid numerical errors associated with too large/small exponential terms,
        # a safer way is to evaluate the r.h.s of Eq.10 as a function of R,
        # find the maximum, and then subtract this maximum before evaluating P(R).
        PROB = np.exp(Exp_index - index_max) * DetQ
        y = PROB/PROB.max() / R # Jefferys 1/R prior
        area = deltaR * np.sum(y)
        y_norm = y / area   # PDF normalization
        max_index = np.argmax(y_norm)
        R_mean = R[max_index]  # find the bestfit R, which means PDF reaches the maximum
        return max_index, R_mean, y_norm

    def pdf_1sigma(max_index, R_mean, y_norm):
        # find the lower limit of R, which requires the integral area is 0.34 (1-sigma)
        f = np.ones(max_index)
        for x in range(0,max_index):
            f[x] = abs(0.34 / deltaR - np.sum(y_norm[x:max_index])) 
        x1 = np.argmin(f)  # the index of lower limit R

        # find the upper limit of R, which requires the integral area is 0.34 (1-sigma)
        v = np.ones(NR - max_index)
        for u in range(max_index,NR):
            v[u-max_index] = abs(0.34 / deltaR - np.sum(y_norm[max_index:u]))
        x2 = max_index + np.argmin(v) # the index of upper limit R

        delta_x1 = R[max_index] - R[x1]
        delta_x2 = R[x2] - R[max_index]
        return x1, x2, delta_x1, delta_x2
    
    def gaussian_func(mu,sigma):
        gau = 1./np.sqrt(2*np.pi) * np.exp(-(R-mu)**2 / 2./sigma**2)
        return gau
    
    max_index, R_mean, y_norm = pdf(d1,d2,cov1,cov2,R)
    x1, x2, delta_x1, delta_x2 = pdf_1sigma(max_index, R_mean, y_norm)
    
    delta_x = (delta_x1 + delta_x2) / 2
    gau = gaussian_func(R_mean, delta_x)
    sum_gau = np.sum(gau)*deltaR
    gau_norm = gau / sum_gau
    
    return R, R_mean, y_norm, x1, x2, max_index, delta_x1, delta_x2, gau_norm
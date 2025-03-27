import numpy as np
import scipy.optimize as opt




def getFairnessProblem(n_g,n_z):
    B = np.random.rand(n_g,n_z)
    B = np.diag(1.0 / np.sum(B, 1))@B
    w = np.random.rand(n_g)
    return B, w

def etta(a,b):
    eps = 1e-13
    return (b <= a)*(1.0 - b / (a + eps)) + (b > a)*(1.0 - (1.0 - b) / (1.0 - a + eps))

def etta_a_deriv(a,b):
    eps = 1e-13
    return (a >= b)*(b/(eps + a**2)) + (a < b)*(-(1.0-b)/(eps + (1.0-a)**2))

def etta_approx(a,b):
    c = etta(a, b)
    g = etta_a_deriv(a, b)
    return c, g

def FairObj(alpha, w, B):
    ng, nz = np.shape(B)
    f = np.zeros(ng)
    for g in range(ng):
        t = etta(alpha, B[g, :])
        f[g] = np.max(t)
    obj = np.dot(f, w)
    return obj, f

def getFairnessApprox(alpha,B):
    ng,nz = np.shape(B)
    (etta0, grad) = etta_approx(alpha, B[0, :])
    M2 = np.diag(grad)
    v = etta0
    for g in range(1,ng):
        (etta0, grad) = etta_approx(alpha, B[g, :])
        M2 = np.vstack((M2,np.diag(grad)))
        v = np.hstack((v,etta0))
    return M2,v

def Linesearch(x_k, e,f_prev,w,B, lr0=1.0,beta=0.5,num_ls=25):
    lr = lr0
    for jj in range(num_ls):
        x_t = x_k + lr * e
        x_t = x_t / sum(x_t)
        fobj_t, ff_t = FairObj(x_t, w, B)
        if fobj_t < f_prev:
            break
        else:
            lr = lr * beta
    return lr



def solveFairness(alpha_0, B,w,tol = 1e-5):
    n_g, n_z = np.shape(B)
    max_step = 1e-1
    w_k = np.hstack((np.zeros(n_z), w))

    ### Setting box bounds
    bounds = []
    pad = 0.0
    for k in range(n_z):
        bounds = bounds + [(0.0+pad,1.0-pad)] # bounds for a
    for k in range(n_z,n_z+n_g):
        bounds = bounds + [(0.0, 1.0)]
    ###########################################
    Aeq = np.hstack((np.ones(n_z), np.zeros(n_g))).reshape(1,n_g + n_z)
    beq = np.ones(1)
    M1 = -(np.kron(np.diag(np.ones(n_g)),np.ones(n_z))).T
    numIter = 200
    fobj = np.zeros(numIter)
    fobj[0], ff = FairObj(alpha_0, w, B)
    x_k = np.hstack((alpha_0, ff))
    for k in range(1,numIter):
        M2, v = getFairnessApprox(x_k[0:n_z], B)
        Aieq_k = np.hstack((M2, M1))
        bieq_k = -v + M2 @ x_k[0:n_z]
        res = opt.linprog(w_k, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq, bounds=bounds,method='revised simplex')
        x_new = res.get("x")
        e = x_new - x_k
        if max(abs(e)) < tol:
            fobj = fobj[0:k]
            break
        if max(abs(e)) > max_step:
            e = max_step * (e / max(abs(e)))
        lr = Linesearch(x_k[0:n_z], e[0:n_z], fobj[k-1], w, B, lr0=1.0, beta=0.5, num_ls=25)
        x_k = x_k + lr*e
        x_k[0:n_z] = x_k[0:n_z] / sum(x_k[0:n_z])
        alpha = x_k[0:n_z]
        fobj[k], ff = FairObj(alpha, w, B)

    alpha = x_k[0:n_z]

    return alpha, fobj

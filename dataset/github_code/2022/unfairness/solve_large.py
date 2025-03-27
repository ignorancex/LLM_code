import numpy as np
import scipy.optimize as opt
from scipy.linalg import block_diag

class LargeProbParam:
    def __init__(self, ng,nz, w,Pi,P,beta_param):
        self.ng = ng
        self.nz = nz
        self.w = w
        self.Pi = Pi
        self.P = P
        self.beta_param = beta_param
        self.wy = np.diag(w)@Pi
        self.wy = self.wy.T

def getInitialGuessFairnessProblem(n_g,n_z):
    alpha = np.random.rand(n_z, n_z)
    alpha = np.diag(1.0 / np.sum(alpha, 1)) @ alpha
    B = np.random.rand(n_z, n_g, n_z)
    for y in range(0,n_z):
        B[y] = np.diag(1.0 / np.sum(B[y], 1)) @ B[y]
    return alpha,B

def etta(a,b,type=0):
    eps = 1e-15
    if type == 1:
        return 1.0 - b / (a + eps)
    elif type == 2:
        return 1.0 - (1.0 - b) / (1.0 - a + eps)
    return (b <= a)*(1.0 - b / (a + eps)) + (b > a)*(1.0 - (1.0 - b) / (1.0 - a + eps))


def etta_a_deriv(a,b,type=0):
    eps = 1e-15
    if type == 1:
        return b/((a+eps)**2)
    elif type == 2:
        return -(1.0-b)/((1.0-a + eps)**2)
    return (a > b)*(b/((a+eps)**2)) + (a < b)*(-(1.0-b)/((1.0-a + eps)**2))

def etta_b_deriv(a,b,type=0):
    eps = 1e-15
    if type == 1:
        return -1/(eps + a)
    elif type ==2:
        return 1.0/(eps+1.0-a)
    return (a >= b)*(-1/(eps + a)) + (a < b)*(1.0/(eps+1.0-a))

def etta_approx(a,b,type=0):
    c = etta(a, b,type)
    ga = etta_a_deriv(a, b,type)
    gb = etta_b_deriv(a, b,type)
    return c, ga,gb


def FairObjLarge(alpha, B, wy, beta_param):
    nz = alpha.shape[0]
    ng = B.shape[1]
    obj = 0.0
    ff = np.zeros((nz,ng))
    for y in range(nz):
        B_yy = B[y,:,y]
        obj_y,ff[y] = FairObjSmall(alpha[y], wy[y], B[y])
        obj = obj + beta_param*obj_y + (1-beta_param)*np.dot(wy[y],1.0-B_yy)

    return obj,ff

def FairObjSmall(alpha, w, B):
    ng, nz = np.shape(B)
    f = np.zeros(ng)
    for g in range(ng):
        t = etta(alpha, B[g, :])
        f[g] = np.max(t)
    obj = np.dot(f, w)
    return obj, f

def seperateIterateIntoComponents(x,nz,ng):
    bitmap = np.zeros(nz*(nz+nz*ng+ng),dtype = bool)
    recond_size = (nz+nz*ng+ng)
    for k in range(0,nz):
        bitmap[(k*recond_size):(k*recond_size+nz)] = True
    alpha = x[bitmap]
    alpha = np.reshape(alpha,(nz,nz))
    bitmap[:] = False
    for k in range(0,nz):
        bitmap[(k*recond_size+nz):(k*recond_size + nz + nz * ng)] = True
    B = x[bitmap]
    B = np.reshape(B,(nz,ng,nz))
    bitmap[:] = False
    for k in range(0, nz):
        bitmap[(k * recond_size + nz + nz * ng):(k * recond_size + nz + nz * ng + ng)] = True
    c = x[bitmap]
    c = np.reshape(c,(nz,ng))
    return alpha,B,c

def combineComponentsIntoIterate(alpha,B,c,nz,ng):
    recond_size = (nz + nz * ng + ng)
    x = np.zeros(nz*recond_size)
    for k in range(0, nz):
        x[(k*recond_size):(k*recond_size+nz)] = alpha[k]
        x[(k*recond_size+nz):(k*recond_size + nz + nz * ng)] = B[k].flatten()
        x[(k*recond_size + nz + nz * ng):(k*recond_size + nz + nz * ng + ng)] = c[k]
    return x

def getFairnessApproxLargeV0(alpha,B,addEtta = True,type=0):
    nz = B.shape[0]
    ng = B.shape[1]
    thresh = -1.0
    M1 = -(np.kron(np.diag(np.ones(ng)), np.ones(nz))).T
    [M2,M3,vy] = getFairnessApproxSmall(alpha[0], B[0],type)
    M2 = M2[vy >= thresh, :]
    M3 = M3[vy >= thresh, :]
    M1t = M1[vy >= thresh, :]
    v = vy[vy >= thresh]
    M = np.hstack((M2, M3 ,M1t)) # this means that the order of variables is alpha, B, c.
    if addEtta:
        v = v - M2 @ alpha[0] - M3 @ (B[0].flatten())
    for y in range(1,nz):
        [M2, M3, vy] = getFairnessApproxSmall(alpha[y], B[y],type)

        M2 = M2[vy >= thresh, :]
        M3 = M3[vy >= thresh, :]
        M1t = M1[vy >= thresh, :]
        vy = vy[vy >= thresh]
        M = block_diag(M,np.hstack((M2, M3 ,M1t)))
        t = vy
        if addEtta:
            t = t - M2@alpha[y] - M3@(B[y].flatten())
        v = np.hstack((v,t))
    return M,v



def getFairnessApproxLargeV2(alpha,B,addEtta = True):
    M1, v1 = getFairnessApproxLargeV0(alpha,B,addEtta,type=1)
    M2, v2 = getFairnessApproxLargeV0(alpha, B, addEtta, type=2)
    M = np.vstack((M1,M2))
    v = np.hstack((v1,v2))
    return M,v


def getFairnessApproxSmall(alpha,B,type=0):
    ng,nz = np.shape(B)
    (etta0, grad_a, grad_b) = etta_approx(alpha, B[0],type)
    M2 = np.diag(grad_a)
    M3 = np.diag(grad_b)
    v = etta0
    for g in range(1,ng):
        (etta0, grad_a, grad_b) = etta_approx(alpha, B[g],type)
        M2 = np.vstack((M2,np.diag(grad_a)))
        M3 = block_diag(M3,np.diag(grad_b))
        v = np.hstack((v,etta0))
    return M2,M3,v



def getLPComponents(nz,ng,param):
    wy = param.wy
    FB = np.zeros((nz, ng, nz))
    Falpha = np.zeros((nz, nz))
    for y in range(nz):
        FB[y,:,y] = -(1-param.beta_param)*wy[y]
    Fc = wy*param.beta_param
    return Falpha,FB,Fc


def getEqualityConstraintsLarge(nz,ng,Pi,P):
    I_nz = np.diag(np.ones(nz))
    A1 = np.hstack((np.ones(nz), np.zeros(nz*ng + ng))).reshape(1,nz + nz*ng + ng)
    A1 = np.kron(I_nz, A1)
    A2 = np.hstack((np.zeros((ng,nz)),np.kron(np.diag(np.ones(ng)), np.ones(nz)), np.zeros((ng,ng))))
    A2 = np.kron(np.diag(np.ones(nz)), A2)
    A3 = np.zeros((nz*ng,nz*(nz+nz*ng+ng)))
    num_params_y = nz + nz * ng + ng
    for g in range(0, ng):
            for y in range(0, nz):
                offset = num_params_y * y + nz + g * nz
                A3[g * nz:(g * nz + nz), offset:(offset + nz)] = np.diag(Pi[g,y]*np.ones(nz))
    Aeq = np.vstack((A1,A2,A3))
    beq = np.hstack((np.ones(nz),np.ones(nz*ng),P.flatten()))
    return Aeq,beq

def getEqualityConstraintsRHSLarge(nz,ng,P,eps = 0.0):
    Pnew = (1 - eps * P.shape[1]) * P + eps
    beq = np.hstack((np.ones(nz), np.ones(nz * ng), Pnew.flatten()))
    return beq

def f_linesearch(x,param):
    alpha, B, c = seperateIterateIntoComponents(x, param.nz, param.ng)
    return FairObjLarge(alpha, B, param.wy, param.beta_param)



def Linesearch(f,param,x_k, e,f_prev, lr0=1.0,beta=0.5,num_ls=25):
    lr = lr0
    fobj_t = 100000.0
    for jj in range(num_ls):
        x_t = x_k + lr * e
        fobj_t, ff_t = f(x_t,param)
        if fobj_t < f_prev:
            break
        else:
            lr = lr * beta
    return lr,fobj_t

def getBoundsLargeV0(nz,ng,eps):
    bounds = []
    for y in range(nz):
        bounds = bounds + [(eps , 1.0-eps)] * nz  # bounds for a
        bounds = bounds + [(eps , 1.0-eps)] * (nz*ng) # bounds for b
        bounds = bounds + [(0.0 , 1.0)] * ng  # bounds for c
    return bounds

def getBoundsLarge(nz,ng,eps,max_step,alphak,Bk,ck):
    bounds = []
    for y in range(nz):
        ay = alphak[y]
        By = Bk[y]
        cy = ck[y]
        for j in range(nz):# bounds for a
            bounds = bounds + [(max(eps,ay[j]-max_step), min(1.0 - eps , ay[j]+max_step))]
        for i in range(ng): # bounds for b
            Byi = By[i]
            for j in range(nz):
                bounds = bounds + [(max(eps, Byi[j] - max_step), min(1.0 - eps, Byi[j] + max_step))]
        for j in range(ng): # bounds for c
            bounds = bounds + [(max(0,cy[j] - max_step), min(1.0,cy[j] + max_step))]
    return bounds

def solveFairness(param, alpha_0, B_0,tol = 1e-4,epsB = 1e-5, max_step = 2e-1, alg ='highs-ipm' ,verbose=False,numIter=250):
    global __testnum__
    ng = param.ng
    nz = param.nz
    beta_param = param.beta_param


    fobj = np.zeros(numIter)
    alpha = alpha_0
    B = B_0
    fobj[0], c = FairObjLarge(alpha, B, param.wy, beta_param)

    Aeq,beq = getEqualityConstraintsLarge(nz, ng, param.Pi, param.P)
    Falpha,FB,Fc = getLPComponents(nz, ng, param)

    wLP = combineComponentsIntoIterate(Falpha, FB, Fc, nz, ng)
    x_k = combineComponentsIntoIterate(alpha, B, c, nz, ng)
    
    for k in range(1,numIter):
        alpha, B, c = seperateIterateIntoComponents(x_k, nz, ng)
        bounds = []
        max_step = max_step
        Aieq_k, bieq_k = getFairnessApproxLargeV2(alpha, B, addEtta=True)
        xxxx, c = FairObjLarge(alpha, B, param.wy, beta_param)
        x_k = combineComponentsIntoIterate(alpha, B, c, nz, ng)

        if k == 1:
            epsBnew = 0.0
            bounds = getBoundsLargeV0(nz, ng, epsBnew + epsB)
        else:
            epsBnew = 0.0
            bounds = getBoundsLarge(nz,ng,epsBnew +  epsB,max_step,alpha,B,c)

        bieq_k = -bieq_k
        res = opt.linprog(wLP, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq,bounds=bounds, method=alg)
        if res.get("status") == 4:
            bounds = getBoundsLarge(nz, ng, epsBnew + epsB, max_step/10.0, alpha, B, c)
            res = opt.linprog(wLP, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq, bounds=bounds,method=alg)
        if res.get("status") == 4:
             bounds = getBoundsLarge(nz, ng, epsBnew + epsB, max_step/50.0, alpha, B, c)
             res = opt.linprog(wLP, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq, bounds=bounds,method=alg)

        if res.get("status")==4:
            res = opt.linprog(wLP, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex',x0=x_k)
        x_new = res.get("x")
        if x_new is None:
            res = opt.linprog(wLP, A_ub=Aieq_k, b_ub=bieq_k, A_eq=Aeq, b_eq=beq, bounds=bounds,
                              method='simplex')
            print('======================== linprog failed =================')
            exit(1)
        e = x_new - x_k
        alphae, Be, ce = seperateIterateIntoComponents(e, nz, ng)
        step_size = max(abs(alphae.flatten()))
        if verbose:
            print("step size: ",step_size/2,": ",max(abs(alphae.flatten())),",", max(abs(Be.flatten())))
        if k==1:
            lr = 1.0
            f_obj = f_linesearch(x_k + e,param)[0]
        else:
            if step_size > max_step:
                e = max_step * (e / step_size)
            lr,f_obj = Linesearch(f_linesearch,param,x_k, e, fobj[k-1], lr0=1.0, beta=0.5, num_ls=30)

        x_k = x_k + lr*e
        fobj[k] = f_obj
        alpha,B,c = seperateIterateIntoComponents(x_k, nz, ng)

        fobj[k], ff = FairObjLarge(alpha, B,  param.wy, beta_param)
        if k % 10 == 0:
            print("**** Fairness Obj iter " ,k,": ", fobj[k], flush=True)
    return alpha, B, fobj

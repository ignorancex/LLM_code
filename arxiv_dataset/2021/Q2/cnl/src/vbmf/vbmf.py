import numpy as np
import scipy as sp
from numpy import random as random
from scipy import linalg as alg
from scipy import optimize as sp_opt


class VBMF2(object):
    """V = BA^T + sigma*Z"""
    def __init__(self, V):
        #TODO args A,B are only for debug
        super(VBMF2, self).__init__()
        self.V = V
        self.L = V.shape[0]
        self.M = V.shape[1]
        self.H = min(self.L, self.M)
        #Frobenius norm
        norm = alg.norm(V)
        self.norm = norm
        self.sigma = norm / np.sqrt(self.L*self.M)
        self.rank = self.H
    def get_rank_analytically(self):
        """
        See Nakajima pp.131.
        """
        TRANSPOSE = False
        if self.L > self.M:
            self.__init__(self.V.T)
            TRANSPOSE = True
        H = self.H
        L = self.L
        M = self.M
        alpha = float(self.L)/ self.M
        tau = 2.5129*sp.sqrt(alpha)
        U,D,W = np.linalg.svd(self.V)
        #for speed-up
        assert D.shape[0] == self.H
        gamma = []
        for h in range(self.H):
            assert D[h] > 0
            gamma.append(D[h])


        inf_x = (1+ tau )*(1 + alpha/tau)

        def tau_alpha(x):
            return (x - 1 - alpha + sp.sqrt((x- 1 - alpha)**2 - 4 * alpha) ) / 2

        def psi1(t):
            return sp.log(t+1) + alpha * sp.log(t/alpha  + 1) - t

        def psi(x):
            return x -sp.log(x+1e-8) + (x > inf_x )* psi1(tau_alpha(x))

        def Omega(x):
            s = 0
            for h in range(H):
                s += psi(x**2 * gamma[h]**2 / M)
            return s / H

        def Omega_list(v):
            return Omega(v[0])

        def d_psi(x):
            return 1 - 1./(x+1e-8)

        def d_Omega(x):
            s = 0
            for h in range(H):
                s += 2*x*gamma[h]**2/M * d_psi(x**2 * gamma[h]**2 / M)
            return s/H


        #x_inits = [ 1./ self.sigma,   100., 500., 10., 1., 0.1]
        solutions =[]
        print("(get_rank_analytically)Start minimizing Omega...")
        Test = False
        maxiter=1000
        options ={'maxiter':maxiter}
        #options={}
        if Test:
            x_inits = [ 5, 10,15, 20]
            print( "x_inits=", x_inits)
            for x_init in x_inits:
                v = [x_init]
                Otype="Scalar"
                if Otype=="Scalar" :
                    result = sp_opt.minimize_scalar(Omega)
                    solution = result.x.real
                    print( "mimize_scalar:", solution)
                    if abs(solution - x_init) > 1e-2:
                        solutions.append(solution)
                elif Otype=="fmin":
                    result = sp_opt.fmin(Omega_list,v, full_output = True, disp = True)#,maxiter=maxiter)
                    solution = result[0][0]
                    print( "fmin:", solution)
                    if abs(solution - x_init) > 1e-1:
                        solutions.append(solution)
                elif Otype=="Powell":
                    result = sp_opt.minimize(Omega_list, v, method='nelder-mead',options=options)
                    solution = result.x[0]
                    print( "nelder:", solution)
                    if abs(solution - x_init) > 1e-1:
                        solutions.append(solution)
                elif Otype == "fmin_cg":
                    result = sp_opt.fmin_cg(Omega_list,v, fprime=d_Omega,full_output = True, disp = False)#,maxiter=maxiter)
                    solution = result[0][0].real
                    #print( "fmin_cg:", solution)
                    if result[1] <1e+34 and abs(solution - x_init)>1e-1:
                        solutions.append(solution)

                    """
                    result = sp_opt.minimize(Omega_list, v, method='Powell', options={'maxiter':10000})
                    solution = result.x.real
                    print( solution
                    solutions.append(solution)

                    result = sp_opt.minimize(Omega_list, v, method='trust-ncg')
                    solution = result.x.real
                    print( solution
                    solutions.append(solution)
                    """
        else:
            result = sp_opt.minimize_scalar(Omega,bounds=(0, 1e+7), method="bounded", options=options)
            solution = result.x.real
            solutions.append(solution)

        #print( "solutions=", solutions)
        if len(solutions) > 0:
            sigmas = [ 1./ abs(s) for s in solutions]
            #import pdb; pdb.set_trace()
            print( "sigmas=",sigmas)
            self.sigma = min(sigmas)
            print( "estimated sigma = ", self.sigma)

            inf_gamma = self.sigma*sp.sqrt(M * inf_x)
            #print( "gamma thres = ", inf_gamma)
            rank = 0
            #post_ratios = []

            for h in range(self.H):
                if gamma[h] > inf_gamma or gamma[h] == inf_gamma:
                    rank += 1
                    #temp =  (self.sigma/gamma[h])**2
                    #temp *= self.M + self.L
                    #temp = 1- temp
                    #post_ratio =  temp
                    #post_ratio += sp.sqrt(temp**2 - 4*self.L*self.M* (self.sigma/gamma[h])**4)
                    #post_ratio *= 0.5
                    #assert post_ratio > 0
                    #post_ratios.append(post_ratio)
        else:
            rank = self.H
            #post_ratios=np.ones(self.H).tolist()
        if TRANSPOSE:
            self.__init__(self.V.T)

        print( "rank=", rank)

        return rank#, post_ratios


class VBMF(object):
    """Variational Bayesian Matrix Factorization"""
    """V = BA^T + Z"""
    def __init__(self, V):
        #TODO args A,B are only for debug
        super(VBMF, self).__init__()
        self.V = V
        self.L = V.shape[0]
        self.M = V.shape[1]
        self.H = min(self.L, self.M)
        #Frobenius norm
        norm = alg.norm(V)
        self.sigma = norm / np.sqrt(self.L*self.M)
        self.A = self.sigma * random.randn(self.M, self.H)
        self.B = self.sigma * random.randn(self.L, self.H)
        self.var_A = np.identity(self.H)
        self.var_B = np.identity(self.H)
        self.c_A = np.ones(self.H)
        self.c_B = np.ones(self.H)

    def each_update(self,seed):
        #update params
        if seed ==1:
            self.var_A = self.sigma**2 * alg.inv( np.dot(self.B.T, self.B) + self.L*self.var_B + self.sigma**2 * np.diag(sp.reciprocal(self.c_A)) )
            self.A = (1./ self.sigma**2) * np.dot(self.V.T, np.dot( self.B ,self.var_A) )
        elif seed ==2:
            self.var_B = self.sigma**2 * alg.inv( np.dot(self.A.T, self.A) + self.M*self.var_A + self.sigma**2 * np.diag(sp.reciprocal(self.c_B)) )
            self.B = (1./ self.sigma**2) * np.dot(self.V, np.dot( self.A , self.var_B))

        #update hyper params
        elif seed ==3:
            for h in range(self.H):
                self.c_A[h]  = alg.norm(self.A.T[h])**2 / self.M + self.var_A[h][h]
                self.c_B[h]  = alg.norm(self.B.T[h])**2 / self.L + self.var_B[h][h]
        elif seed ==4:
            #update variance of noize
            ATA = np.dot(self.A.T, self.A)
            BTB = np.dot(self.B.T, self.B)
            BAT = np.dot(self.B, self.A.T)
            traced_1 = 2*np.dot(self.V.T, BAT)
            traced_2 = np.dot(ATA + self.M*self.var_A, BTB + self.L * self.var_B)
            sig2 = alg.norm(self.V)**2 - sp.trace(traced_1)+ sp.trace(traced_2)
            sig2 = sig2/ (self.L*self.M)
            self.simga = sp.sqrt(sig2)

    def update(self, RANDOM=False):
        seeds = range(4)
        if RANDOM:
            random.shuffle(seeds)
        for seed in seeds:
            #print(seed)
            self.each_update(seed+1)

    def free_energy(self, eps=1e-8):
        ATA = np.dot(self.A.T, self.A)
        BTB = np.dot(self.B.T, self.B)
        traced =  np.dot(np.diag(sp.reciprocal(self.c_A)), ATA + self.M*self.var_A)
        traced+=  np.dot(np.diag(sp.reciprocal(self.c_B)), BTB + self.L*self.var_B)
        traced+=  (-np.dot(ATA, BTB) + np.dot(ATA + self.M*self.var_A, BTB + self.L*self.var_B))/ self.sigma**2

        f  = self.L * self.M * sp.log(2*sp.pi*self.sigma**2)
        f += alg.norm(self.V - np.dot(self.B, self.A.T))**2 / self.sigma**2
        f += self.M * ( sp.log(self.c_A.prod()+eps)-sp.log(alg.det(self.var_A)+eps))
        f += self.L * ( sp.log(self.c_B.prod()+eps)-sp.log(alg.det(self.var_B)+eps))
        f -= (self.L + self.M)*self.H
        f += sp.trace(traced)

        return f

    def optimize(self,threshold, max_iter, test_interval, use_decreament=False, RANDOM=True):
        assert max_iter  > 0
        assert threshold > 0
        assert test_interval > 0
        f = self.free_energy()

        norm =  alg.norm( np.dot(self.B, self.A.T))
        text ="Initial Free_energy = {0:.5f}".format(f)
        text+=" : ||BA^T||_Fro= {0:.5f}".format(norm)
        text+=" : sigma= {0:.5f}".format( self.sigma)
        print(text)

        for m in range(max_iter):
            self.update(RANDOM)
            if m % test_interval == 0:
                f_temp = self.free_energy()
                decrement = -1
                if use_decreament:
                    decrement = abs(f - f_temp)/ test_interval
                f = f_temp
                norm =  alg.norm( np.dot(self.B, self.A.T))
                text ="Iteration {0:07d}:".format(m)
                text +=" : Energy = {0:.5f}".format(f_temp)
                if use_decreament:
                    text+=" : Decreament = {0:.5f}\n".format(decrement)
                print(text)
                text=" : ||BA^T||_Fro= {0:.5f}".format(norm)
                Z = self.sigma*random.randn(self.L,self.M)
                norm =  alg.norm( self.V - np.dot(self.B, self.A.T)-Z)/sp.sqrt(self.M*self.L)
                text+=" : ||V-BA^T-Z||_Fro/sqrt(LM)= {0:.5f}".format(norm)

                text+=" : sigma= {0:.5f}".format( self.sigma)
                print(text)
                ok = (norm > 0)
                if use_decreament:
                    ok = (ok and decrement > 0 and decrement < 1e+20 )
                if not ok:
                    print("Missing! Redo...")
                    self.__init__(self.V)
                    f = self.free_energy()
                if use_decreament:
                    if decrement < threshold:
                        print("Optimization Done.")
                        return
        print("Reach max_iteration.")
        return

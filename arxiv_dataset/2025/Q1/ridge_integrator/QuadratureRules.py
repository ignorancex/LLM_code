#%% import modules

import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.special
from scipy.interpolate import Rbf
import cProfile
import math
from scipy.optimize import linprog
from scipy.optimize import minimize
import scipy.optimize

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds." )
    else:
        print( "Toc: start time not set" )

#%% QuadratureRules - Super class

class QuadratureRules(object):
    
    def __init__(self, Dimensions = 1):
        self.Dimensions = Dimensions
        self.Bounds = np.array([np.zeros(self.Dimensions),
                                np.ones(self.Dimensions)])
        self.Abscissas = []
        self.Weights = []
        self.Kernel = None
        return
    
    def Integrate(self, fun, ListIndex = None):
        i = 0 if ListIndex is None else ListIndex
        if type(fun) is np.ndarray:
            return np.dot(self.Weights[i],fun)
        else:
            return np.dot(self.Weights[i],fun(self.Abscissas[i]))
    
    def set_Weights(self, Weights):
        self.Weights = Weights
        return
    
    def set_Abscissas(self, Abscissas):
        self.Abscissas = Abscissas
        return
        
    def set_Abscissas_and_Weights(self, Abscissas = None, Weights = None):
        if Abscissas is not None:
            self.Abscissas = Abscissas
        if Weights is not None:
            self.Weights = Weights
        return

    def save_to_file(self,filename,comment=''):
        skip = '' if comment == '' else '\n'
        w = self.Weights.reshape((len(self.Weights),1)) if len(self.Weights.shape) == 1 else self.Weights
        x = np.concatenate((self.Abscissas,w),axis=1)
        np.savetxt(filename,x,header=comment + skip + 
                   str(self.Abscissas.shape[1]) + ' of rows Abscissas followed by Weights')
        return
        
    def load_from_file(self,filename,Dimensions=None):
        if Dimensions is None: Dimensions = self.Dimensions
        x = np.loadtxt(filename)
        self.Abscissas = x[:,0:Dimensions]
        w = x[:,Dimensions:None]
        self.Weights = w.reshape((len(w))) if w.shape[1] == 1 else w
        return
    
    def save_to_dict(self,filename):
        D = {'Weights': self.Weights, 'Abscissas': self.Abscissas}
        np.save(filename,D)
        return
    
    def load_from_dict(self,filename):
        D = np.load(filename,allow_pickle='TRUE').item()
        self.Weights = D['Weights']
        self.Abscissas = D['Abscissas']
        return

#%% Univariate Quadrature Rules

class Univariate_Quadrature(QuadratureRules):
    
    def __init_(self):
        QuadratureRules.__init__(self)
        return
        
    def get_quadrature(self, typ, NumberofPoints):
        if typ[0].lower() == 'l': # gauss legendre rule
            p, w = np.polynomial.legendre.leggauss(NumberofPoints)
            self.Abscissas.append( (p+1) / 2 )
            self.Weights.append( w / 2 )
            
        elif typ[0].lower() == 't': # iterated trapezoidial rule
            if NumberofPoints > 1:
                p = np.linspace(0,1,NumberofPoints)
                w = np.ones(NumberofPoints) / (NumberofPoints - 1)
                w[0] = w[0] / 2
                w[-1] = w[-1] / 2
            else:
                p = np.array([0.5])
                w = np.array([1.])
            self.Abscissas.append( p )
            self.Weights.append( w )
            
        elif typ[0].lower() == 'c': # clenshaw curtis rule
            if NumberofPoints > 1:
                n = NumberofPoints
                
                theta = (n - 1 - np.arange(n))*np.pi / (n - 1)
                
                p = np.cos( theta )
                p[0] = -1
                p[-1] = 1
                if np.mod(n,2) == 1:
                    p[ (n-1)//2 ] = 0
                    
                w = np.ones( n )            
                for i in range(n):
                  for j in range( (n-1)//2 ):
                    if ( 2*(j+1) == (n-1) ):
                      b = 1.0
                    else:
                      b = 2.0
                    w[i] = w[i] - b * np.cos( 2*(j+1)*theta[i] ) / ( 4 * j * ( j + 2 ) + 3 )
            
                w[0] = w[0] / (n-1)
                for i in range(1,n-1):
                  w[i] = 2*w[i] / (n-1)
                w[n-1] = w[n-1] / (n-1)
            elif NumberofPoints == 1:
                p = np.array([0])
                w = np.array([2])
            self.Abscissas.append( (p+1) / 2 )
            self.Weights.append( w / 2 )
            
        return


#%% TensorProduct Quadrature

class TensorProduct_Quadrature(QuadratureRules):
    
    def __init__(self,Dimensions = 1):
        self.UnivariateBasis = None
        QuadratureRules.__init__(self, Dimensions=Dimensions)
        return
        
    def set_UnivariateBasis(self, types = ['l'], NumberofPoints = [5], abscissas = None, weights = None):
        self.UnivariateBasis = {'Abscissas': [], 'Weights': [], 'NumberofPoints': []}
        if (abscissas is None) or (weights is None):
            if (len(types) not in [1,self.Dimensions]) or (len(NumberofPoints) not in [1,self.Dimensions]):
                TypeError('types and NumberofPoints must be a list of length 1 or self.Dimensions')
            for typ,n in zip(types,NumberofPoints):
                QF = Univariate_Quadrature()
                QF.get_quadrature(typ,n)
                self.UnivariateBasis['Abscissas'].append( QF.Abscissas[0] )
                self.UnivariateBasis['Weights'].append( QF.Weights[0] )
                self.UnivariateBasis['NumberofPoints'].append( n )
        else:
            if (len(abscissas) not in [1,self.Dimensions]) or (len(weights) not in [1,self.Dimensions]):
                TypeError('abscissas and weights must be a list of length 1 or self.Dimensions')
            for p,w in zip(abscissas,weights):
                self.UnivariateBasis['Abscissas'].append( p )
                self.UnivariateBasis['Weights'].append( w )
                self.UnivariateBasis['NumberofPoints'].append( len(w) )
                
        return
    
    def get_Tensor_Quadrature(self):
        if len(self.UnivariateBasis['Weights']) == 1:
            repeat = self.Dimensions
        else:
            repeat = 1
        
        self.Abscissas.append( np.array(list(itertools.product(*self.UnivariateBasis['Abscissas'],repeat=repeat))) )
        self.Weights.append( np.prod(np.array(list(itertools.product(*self.UnivariateBasis['Weights'],repeat=repeat))),axis=1) )
        
        return
    
    def get_Tensor_Quadrature_Sequence(self, typ = 'l', NumberofPoints = 5):
        for n in range(NumberofPoints,0,-1):
            self.set_UnivariateBasis(types=[typ],NumberofPoints=[n])
            self.get_Tensor_Quadrature()
        return
        


#%% Monte Carlo Quadrature

class MonteCarlo_Quadrature(QuadratureRules):
    
    def __init__(self,Dimensions = 1):
        self.NestStructure = None
        QuadratureRules.__init__(self, Dimensions = Dimensions)
        return
    
    def get_MC_Abscissas_and_Weights(self, NumberofPoints = 1, Sampling_Procedure = 'r',
                                     abscissas_0 = [], threshold = 0, p_max = None, Ip = None, N_Ip = 10000):
        if self.Kernel is None:
            abscissas = np.random.rand(NumberofPoints-len(abscissas_0),self.Dimensions)
            abscissas = np.concatenate((abscissas,abscissas_0), axis = 0) if len(abscissas_0) > 0 else abscissas
            weights = np.ones(NumberofPoints) / NumberofPoints
                            
        elif callable(self.Kernel):
            weights = np.zeros(NumberofPoints)
            if Sampling_Procedure == 'r':
                abscissas = np.zeros([NumberofPoints-len(abscissas_0),self.Dimensions])
                abscissas = np.concatenate((abscissas,abscissas_0), axis = 0) if len(abscissas_0) > 0 else abscissas
                if Ip is None or p_max is None:
                    p = self.Kernel(np.random.rand(N_Ip,self.Dimensions))
                Ip = np.sum(p) / N_Ip if Ip is None else Ip
                p_max = p.max() if p_max is None else p_max

                i = 0
                count = 0
                print('obtaining new Abscissas / #points',NumberofPoints-len(abscissas_0))
                while i < NumberofPoints-len(abscissas_0):
                    count += 1
                    abscissas_proposal = np.random.rand(1,self.Dimensions)
                    tmp = self.Kernel(abscissas_proposal) / p_max
                    if np.random.rand() < tmp and tmp > threshold:
                        abscissas[i] = abscissas_proposal
                        weights[i] = Ip /(self.Kernel(abscissas_proposal) * NumberofPoints)
                        i += 1
                print('done with',count,'point evaluations for',NumberofPoints-len(abscissas_0),'new abscissas')
                for i in range(len(abscissas_0)):
                    weights[NumberofPoints-len(abscissas_0)+i] = Ip /(self.Kernel(abscissas_0[i]) * NumberofPoints)
                    
        else:
            print('no such option available')
        self.Abscissas.append( abscissas )
        self.Weights.append( weights )
        
        L = np.zeros(NumberofPoints, dtype=bool)
        L[NumberofPoints-len(abscissas_0):NumberofPoints] = True
            
        return L
    
    def get_MC_Sequence(self, NumberofPoints = 2**np.arange(10), nested = False, threshold = 0,
                        p_max = None, Ip = None, N_Ip = 10000):
        NumberofPoints = 2**np.arange(NumberofPoints) if len(np.array(NumberofPoints).shape) == 0 else NumberofPoints
        if self.Kernel is not None:
            if Ip is None or p_max is None:
                p = self.Kernel(np.random.rand(N_Ip,self.Dimensions))
            Ip = np.sum(p) / N_Ip if Ip is None else Ip
            p_max = p.max() if p_max is None else p_max
            
        if not nested:
            for n in np.flipud(NumberofPoints):
                self.get_MC_Abscissas_and_Weights(NumberofPoints=n, threshold=threshold, p_max = p_max, Ip = Ip, N_Ip = N_Ip)
        else:
            NestStructure = []
            abscissas_0 = []
            for n in NumberofPoints:
                L = self.get_MC_Abscissas_and_Weights(NumberofPoints=n,abscissas_0=abscissas_0,threshold=threshold,
                                                      p_max = p_max, Ip = Ip, N_Ip = N_Ip)
                abscissas_0 = self.Abscissas[-1]
                NestStructure.append( L )
            self.NestStructure = NestStructure
            self.NestStructure.reverse()
            self.Abscissas.reverse()
            self.Weights.reverse()
        return
    

#%% Smolyak Quadrature

class Smolyak_Quadrature(QuadratureRules):
    
    def __init__(self, Dimensions = 1):
        self.UnivariateQuadratureSequence = None
        QuadratureRules.__init__(self,Dimensions = Dimensions)
        return
    
    def set_UnivariateQuadratureSequence(self, typ = ['l'], NumberofPoints = None, abscissas = None, weights = None):
        self.UnivariateQuadratureSequence = {'Abscissas': [], 'Weights': [], 'NumberofPoints': []}
        if NumberofPoints is not None:
            for n in NumberofPoints:
                QF = Univariate_Quadrature()
                QF.get_quadrature(typ[0], n)
                self.UnivariateQuadratureSequence['Abscissas'].append( QF.Abscissas )
                self.UnivariateQuadratureSequence['Weights'].append( QF.Weights )
                self.UnivariateQuadratureSequence['NumberofPoints'].append( n )
        else:
            for p,w in zip(abscissas,weights):
                self.UnivariateQuadratureSequence['Abscissas'].append( p )
                self.UnivariateQuadratureSequence['Weights'].append( w )
                self.UnivariateQuadratureSequence['NumberofPoints'].append( len(w) ) 
        return
    
    def get_Smolyak_Quadrature(self, Level = None, nested = False):
        if Level is None:
            Level = len(self.UnivariateQuadratureSequence['NumberofPoints'])
            
        alphas = np.array(list(itertools.product(np.arange(Level+1),repeat=self.Dimensions)))
        alphas = alphas[ np.logical_and(np.sum(alphas,axis=1) <= Level, np.sum(alphas,axis=1) >= Level - self.Dimensions - 1) ,:]
        
        abscissas = []
        weights = []
        for alpha in alphas:
            abscissas_proposal = list(np.array(list( itertools.product( *[self.UnivariateQuadratureSequence['Abscissas'][alpha[i]] for i in range(self.Dimensions)] ) )))
            
            weights_proposal = (-1)**(Level-sum(alpha)) * \
                scipy.special.binom(self.Dimensions-1,Level-sum(alpha)) * \
                np.prod( np.array(list( itertools.product( *[self.UnivariateQuadratureSequence['Weights'][alpha[i]] for i in range(self.Dimensions)] ) )), axis=1 )
            weights_proposal = list(weights_proposal)
            
            if nested:
                for i in range(len(weights)):
                    for j in range(len(weights_proposal)-1,-1,-1):
                        distance = sum(np.abs( abscissas[i] - abscissas_proposal[j] ))
                        if distance < 1e-10:
                            weights[i] += weights_proposal[j]
                            del weights_proposal[j]
                            del abscissas_proposal[j]
                    
            for p,w in zip(abscissas_proposal,weights_proposal):
                abscissas.append( p )
                weights.append( w )
        
        abscissas = np.array(abscissas)
        weights = np.array(weights)
        
        L = np.abs(weights) > 1e-10
        if sum(weights[np.logical_not(L)]) > 1e-5: print('Warning: discarded weights are non negegible')
        abscissas = abscissas[L]
        weights = weights[L]
        weights = weights / sum(weights)
        
        self.Abscissas.append( abscissas )
        self.Weights.append( weights )
                    
        return
    
    
#%% functions for rbf evaluation

def eval_rbf_Kernel(x,As,cetas,mus):
    x = x.reshape(1,mus.shape[1]) if len(x.shape) == 1 else x
    return np.array([np.sum(As*np.exp(-np.sum((cetas[None,:]*(x[i][None,:]-mus))**2,axis = 1))) for i in range(len(x))])
        
    
#%% L1 Quadrature

class L1_Quadrature(QuadratureRules):
    
    def __init__(self, Dimensions = 1):
        self.AbscissasProposal = None
        self.MaximumPolynomialDegree = None
        self.PolynomialBasis = None
        self.RHS = None
        self.NestStructure = None
        QuadratureRules.__init__(self, Dimensions = Dimensions)
        self.Weights = []
        self.Abscissas = []
        return
    
    def set_PolynomialBasis(self, MaximumDegree = 5, orthogonalize = False):
        self.MaximumPolynomialDegree = MaximumDegree
        self.PolynomialBasis = ndPolynomial(Dimensions=self.Dimensions, maxdeg=self.MaximumPolynomialDegree)
        self.PolynomialBasis.Kernel = self.Kernel
        return
        
    def get_Random_AbscissasProposal(self, NumberofPoints = 1, Sampling_Procedure = 'r',
                                     use_Kernel = True):
        if (self.Kernel is None) or (use_Kernel is False):
            p = np.random.rand(NumberofPoints,self.Dimensions)            
                        
        elif type(self.Kernel) is list:
            if Sampling_Procedure == 'r':
                p = np.zeros([NumberofPoints,self.Dimensions])
                i = 0
                while i < NumberofPoints:
                    p_proposal = np.random.rand(self.Dimensions)
                    if np.random.rand() < eval_rbf_Kernel(p_proposal,*self.Kernel):
                        p[i] = p_proposal
                        i += 1
                        
        elif callable(self.Kernel):
            if Sampling_Procedure == 'r':
                p = np.zeros([NumberofPoints,self.Dimensions])
                i = 0
                while i < NumberofPoints:
                    p_proposal = np.random.rand(1,self.Dimensions)
                    if np.random.rand() < self.Kernel(p_proposal):
                        p[i] = p_proposal
                        i += 1
        else:
            print('no such option available')
        self.AbscissasProposal = p
        return

    def get_RHS(self, monomial = True, analytic = False, N_points_1d = 10):
        self.PolynomialBasis.get_MonomIntegrals(analytic = analytic, N_points_1d = N_points_1d)
        if monomial:
            self.RHS = self.PolynomialBasis.MonomIntegrals
        else:
            self.RHS = self.PolynomialBasis.get_Integrals()
        return

    def get_L1_Quadrature(self, NumberofBasisFunctios = None, monomial = True):
        n = len(self.PolynomialBasis.MonomialBasis) if NumberofBasisFunctios is None else NumberofBasisFunctios
        rhs = self.RHS[0:n]
        m = self.AbscissasProposal.shape[0]
        if monomial:
            B = self.PolynomialBasis.evaluate_MonomialBasis(self.AbscissasProposal,NumberofBasisFunctios=n)
        else:
            print('still to do')
            #B = self.PolynomialBasis.evaluate(self.AbscissasProposal)
        
        output_linprog = linprog(np.ones(2*m),A_eq=np.concatenate((B,-B),axis=1),
                                 b_eq=rhs,bounds=(0,None),method='highs-ipm')
        
        if not output_linprog.success:
            print('there appears to be a problem in the linear program: ' + output_linprog['message'])
            w = np.array([0])
            p = np.zeros((1,self.Dimensions))
            L = np.array([1],dtype=bool)
        else:
            w = output_linprog.x
            w = w[0:m] - w[m:2*m]
            L = np.abs(w) > 1e-10
            w = w[L]
            p = self.AbscissasProposal[L]
            Lm = w < 0
            print( len(w), 'weights /', sum(Lm),'negative weights:', sum(w[Lm]),'/',sum(w),'/',-sum(w[Lm]) / sum(w))
    
        self.Weights.append(w)
        self.Abscissas.append(p)
    
        return output_linprog, L
        

    def get_nested_L1_Quadrature_Sequence(self, maxdeg = None):
        maxdeg = self.MaximumPolynomialDegree if maxdeg is None else maxdeg
        self.NestStructure = []
        for i in range(maxdeg,-1,-1):
            n = get_Number_of_Polynomials_below_Degree(self.Dimensions, i)
            _, L = self.get_L1_Quadrature(NumberofBasisFunctios=n)
            self.AbscissasProposal = self.Abscissas[-1]
            self.NestStructure.append( L )
        self.NestStructure.append( np.zeros(len(self.Abscissas[-1]),dtype=bool) )
        del self.NestStructure[0]
        return


#%% L1 class with ndBasis class

class L1_Quadrature_2(QuadratureRules):
    
    def __init__(self, Dimensions = 1):
        self.AbscissasProposal = None
        self.MaximumDegree = None
        self.Basis = None
        self.RHS = None
        self.NestStructure = None
        QuadratureRules.__init__(self, Dimensions = Dimensions)
        self.Weights = []
        self.Abscissas = []
        return
    
    def set_Basis(self, maxdeg = 5, maxdeg_factor_vec = None, basis_type = None, 
                 sym_ops = [], periodicity_factor = None, ranges = None):
        self.MaximumDegree = maxdeg
        self.Basis = ndBasis(dim = self.Dimensions, maxdeg = self.MaximumDegree,
                             maxdeg_factor_vec = maxdeg_factor_vec, basis_type = basis_type, 
                             sym_ops = sym_ops, periodicity_factor = periodicity_factor,
                             ranges = ranges)
        self.Basis.Kernel = self.Kernel
        return
        
    def get_Random_AbscissasProposal(self, NumberofPoints = 1, Sampling_Procedure = 'r',
                                     use_Kernel = True):
        if (self.Kernel is None) or (use_Kernel is False):
            p = np.random.rand(NumberofPoints,self.Dimensions)            
                                                
        elif callable(self.Kernel):
            if Sampling_Procedure == 'r':
                p = np.zeros([NumberofPoints,self.Dimensions])
                i = 0
                while i < NumberofPoints:
                    p_proposal = np.random.rand(1,self.Dimensions)
                    if np.random.rand() < self.Kernel(p_proposal):
                        p[i] = p_proposal
                        i += 1
        else:
            print('no such option available')
        self.AbscissasProposal = p
        return

    def get_RHS(self, analytic = False, N_points_1d = 10):
        self.Basis.get_BasisIntegrals(analytic = analytic, N_points_1d = N_points_1d)
        self.RHS = self.Basis.BasisIntegrals
        return

    def get_L1_Quadrature(self, Degree = -1):
        rhs = self.RHS[self.Basis.basis_indices_below_degree[Degree]]
        m = self.AbscissasProposal.shape[0]
        B = self.Basis.eval_basis_funs(self.AbscissasProposal,maxdeg=Degree)
        output_linprog = linprog(np.ones(2*m),A_eq=np.concatenate((B,-B),axis=1),
                                 b_eq=rhs,bounds=(0,None),method='highs-ipm')
        
        if not output_linprog.success:
            print('there appears to be a problem in the linear program: ' + output_linprog['message'])
            w = np.array([0])
            p = np.zeros((1,self.Dimensions))
            L = np.array([1],dtype=bool)
        else:
            w = output_linprog.x
            w = w[0:m] - w[m:2*m]
            L = np.abs(w) > 1e-10
            w = w[L]
            p = self.AbscissasProposal[L]
            Lm = w < 0
            print( len(w), 'weights /', sum(Lm),'negative weights:', sum(w[Lm]),'/',sum(w),'/',-sum(w[Lm]) / sum(w))
    
        self.Weights.append(w)
        self.Abscissas.append(p)
    
        return output_linprog, L
        

    def get_nested_L1_Quadrature_Sequence(self, maxdeg = None):
        maxdeg = self.MaximumDegree if maxdeg is None else maxdeg
        self.NestStructure = []
        for i in range(maxdeg,-1,-1):
            _, L = self.get_L1_Quadrature(Degree = i)
            self.AbscissasProposal = self.Abscissas[-1]
            if i == maxdeg:
                L = np.ones(len(self.Weights[-1]),dtype=bool)
            self.NestStructure.append( L )
        return  
    

#%% L1D external functions

def greedy_cluster(abscissas, weights, steps = 1):
    if len(weights) == 1: return abscissas, weights
    for i in range(steps):
        weights = weights.reshape((len(weights),1)) if len(weights.shape) == 1 else weights
        i0 = np.argmin(np.sum(np.abs(weights),axis=1))
        x0 = abscissas[i0]
        w0 = weights[i0]
        w0n = np.sum(np.abs(w0))
        
        dist_x0 = np.linalg.norm(abscissas - x0,axis=1)
        dist_x0[i0] = np.max(dist_x0) + 1
        
        i1 = np.argmin(dist_x0)
        x1 = abscissas[i1]
        w1 = weights[i1]
        w1n = np.sum(np.abs(w1))
        
        w = w0 + w1
        wn = np.sum(np.abs(w))
        x = (w0n*x0 + w1n*x1) / wn
        
        L = np.ones(len(weights),dtype=bool)
        L[[i0,i1]] = False
        abscissas_new = np.zeros([len(weights)-1,abscissas.shape[1]])
        abscissas_new[0:-1] = abscissas[L]
        abscissas_new[-1] = x
        weights_new = np.zeros([len(weights)-1,weights.shape[1]])
        weights_new[0:-1] = weights[L]
        weights_new[-1] = w
        
        weights = weights_new
        abscissas = abscissas_new

    return abscissas_new, weights_new

def check_weight_quality(weights):
    w0 = weights[:,0]
    nwd = np.sum( np.abs( weights[:,1:None] ) )
    
    l0 = np.sum(w0 < 1e-10) / len(w0)
    ld = nwd / np.sum(w0)
    lm = np.abs(np.sum(w0[w0<0])) / np.sum(w0)
    
    return l0, ld, lm


#%% L1 Quadrature with Derivative

class L1D_Quadrature(L1_Quadrature):
    
    def __init__(self, Dimensions = 1):
        L1_Quadrature.__init__(self,Dimensions=Dimensions)
        return
    
    def get_L1D_Quadrature(self, NumberofBasisFunctios = None, monomial = True):
        n = len(self.PolynomialBasis.MonomialBasis) if NumberofBasisFunctios is None else NumberofBasisFunctios
        rhs = self.RHS[0:n]
        m = self.AbscissasProposal.shape[0]
        if monomial:
            B = np.zeros([n,(self.Dimensions+1)*m])
            B[:,0:m] = self.PolynomialBasis.evaluate_MonomialBasis(self.AbscissasProposal,NumberofBasisFunctios=n)
            for i in range(self.Dimensions):
                B[:,(i+1)*m:(i+2)*m] = self.PolynomialBasis.evaluate_MonomialBasis_Derivative(i, self.AbscissasProposal, n)
        else:
            print('still to do')
            
        cost_vec = np.ones(2*(self.Dimensions+1)*m)
        Mat_lin = np.concatenate((B,-B),axis=1)
        output_linprog = linprog(cost_vec,A_eq=Mat_lin,b_eq=rhs,bounds=(0,None),
                                 method='highs-ipm')

        if not output_linprog.success:
            print('there appears to be a problem in the linear program: ' + output_linprog['message'])
            ww = np.zeros((1,self.Dimensions))
            p = np.zeros((1,self.Dimensions))
            L = np.array([1],dtype=bool)
        else:
            w = output_linprog.x
            w = w[0:(self.Dimensions+1)*m] - w[(self.Dimensions+1)*m:2*(self.Dimensions+1)*m]
            w[np.abs(w) < 1e-10] = 0
            p = self.AbscissasProposal
            ww = w.reshape((p.shape[1] + 1,p.shape[0])).T
            L = np.sum( np.abs(ww),axis=1 ) > 1e-10
            ww = ww[L]
            p = p[L]

        self.Weights.append(ww)
        self.Abscissas.append(p)
            
        return output_linprog, L
    
    def get_L1D_Quadrature_reduced(self, NumberofBasisFunctios = None, reduction_method='del',
                                   quality_limits=[0.2,0.2,0.05]):
        n = len(self.PolynomialBasis.MonomialBasis) if NumberofBasisFunctios is None else NumberofBasisFunctios
        old_listlen = len(self.Weights)
        boolean = True
        while boolean:
            lin_out, _ = self.get_L1D_Quadrature(NumberofBasisFunctios=n)

            if lin_out.success:
                #print(self.Weights[-1])
                l0, ld, lm = check_weight_quality(self.Weights[-1])
                print(l0,ld,lm)
                if l0 > quality_limits[0] or ld > quality_limits[1] or lm > quality_limits[2]:
                    if len(self.Weights) > old_listlen+1:
                        del self.Weights[-1]
                        del self.Abscissas[-1]
                    else:
                        print('bad weights encountered in 1st iteration')
                    break

            else:
                print('unfeasibility warning can be ignored!')
                if len(self.Weights) > old_listlen+1:
                    del self.Weights[-1]
                    del self.Abscissas[-1]
                else:
                    print('unfeasibility encountered in 1st iteration')
                break
            
            print('reducing')
            if reduction_method == 'del':
                #L = np.sum( np.abs(self.Weights[-1]), axis=1) >= 0.5*np.mean( np.abs( self.Weights[-1][np.abs(self.Weights[-1]) > 0] ) )
                L1 = np.ones(len(self.Weights[-1]),dtype=bool)#self.Weights[-1][:,0] == self.Weights[-1][:,0].min()
                indices = np.arange(len(L1))[L1]
                ind1 = np.argmin(np.sum(np.abs(self.Weights[-1][L1,0:None]),axis=1))
                # print(self.Weights[-1][indices[ind1]])
                boolean = len(self.Weights[-1]) > 1
                self.AbscissasProposal = np.delete(self.Abscissas[-1],indices[ind1],axis=0)
                
            elif reduction_method == 'greedy':
                boolean = len(self.Weights[-1]) != 1
                if not boolean: break
                abscissas_new, weights_new = greedy_cluster(self.Abscissas[-1], self.Weights[-1], steps = 1)
                self.AbscissasProposal = abscissas_new
                
            
        del self.Weights[old_listlen:-1]
        del self.Abscissas[old_listlen:-1]
        self.AbscissasProposal = self.Abscissas[-1]
        print('done')
        return
    
    
    def get_nested_L1D_Quadrature_Sequence(self,maxdeg=None,quality_limits=[0.2,0.2,0.05]):
        maxdeg = self.MaximumPolynomialDegree if maxdeg is None else maxdeg
        old_listlen = len(self.Weights)
        for i in range(maxdeg,-1,-1):
            n = get_Number_of_Polynomials_below_Degree(self.Dimensions, i)
            typ = 'greedy' if i == self.MaximumPolynomialDegree else 'del'
            self.get_L1D_Quadrature_reduced(NumberofBasisFunctios=n,reduction_method=typ,quality_limits=quality_limits)
            self.AbscissasProposal = self.Abscissas[-1]
        
        listlen = len(self.Weights)
        self.NestStructure = []
        for i in range(old_listlen+1,listlen):
            L = np.zeros(len(self.Abscissas[i-1]),dtype=bool)
            for k in range(len(self.Abscissas[i-1])):
                L[k] = self.Abscissas[i-1][k] in self.Abscissas[i]
            self.NestStructure.append( L )
        self.NestStructure.append( np.zeros(len(self.Abscissas[i])) )
        return
    
    def Integrate(self, fun, dfun, ListIndex = None):
        i = -1 if ListIndex is None else ListIndex
        if type(fun) is np.ndarray:
            return np.dot(self.Weights[i][:,0],fun) + sum( np.dot(self.Weights[i][:,j+1],dfun[:,j]) for j in range(self.Dimensions) )
        else:
            df_p = dfun(self.Abscissas[i])
            return np.dot(self.Weights[i][:,0],fun(self.Abscissas[i])) + sum( np.dot(self.Weights[i][:,j+1],df_p[:,j]) for j in range(self.Dimensions) )
        

#%% Reduced Quadrature

class Reduced_Quadrature(L1_Quadrature):
    
    def __init__(self, Dimensions = 1):
        self.L1_Abscissas = None
        self.L1_Weights = None
        self.Objective_and_Derivative = None
        self.Objective = None
        self.Objective_Derivative = None
        L1_Quadrature.__init__(self,Dimensions=Dimensions)
        return
    
    def get_L1_Abscissas_and_Weights(self, NumberofBasisFunctios = None, monomial = True):
        output_linprog, _ = self.get_L1_Quadrature(NumberofBasisFunctios=NumberofBasisFunctios,monomial=monomial)
        while not output_linprog.success:
            print('did not find L1 solution: retry')
            del self.Weights[-1]
            del self.Abscissas[-1]
            self.get_Random_AbscissasProposal(NumberofPoints=len(self.AbscissasProposal),use_Kernel=True)
            output_linprog, _ = self.get_L1_Quadrature(NumberofBasisFunctios=NumberofBasisFunctios,monomial=monomial)
            
        self.L1_Abscissas =  [ self.Abscissas[-1] ]
        self.L1_Weights =  [ np.abs(self.Weights[-1]) ]
        if np.any(self.Weights[-1] < 0): print(sum(self.Weights[-1] < 0),'negative Weights ignored',sum(self.Weights[-1][self.Weights[-1] < 0]))
        del self.Weights[-1]
        del self.Abscissas[-1]
        return
    
    def get_Cluster_Sequence(self, min_size = 1):
        while len(self.L1_Weights[-1]) > min_size:
            i0 = np.argmin(self.L1_Weights[-1])
            x0 = self.L1_Abscissas[-1][i0]
            w0 = self.L1_Weights[-1][i0]
            
            dist_x0 = np.linalg.norm(self.L1_Abscissas[-1] - x0,axis=1)
            dist_x0[i0] = np.max(dist_x0) + 1
            
            i1 = np.argmin(dist_x0)
            x1 = self.L1_Abscissas[-1][i1]
            w1 = self.L1_Weights[-1][i1]
            
            w = w0 + w1
            x = (w0*x0 + w1*x1) / w
            
            L = np.ones(len(self.L1_Weights[-1]),dtype=bool)
            L[[i0,i1]] = False
            abscissas = np.zeros([len(self.L1_Weights[-1])-1,self.Dimensions])
            abscissas[0:-1] = self.L1_Abscissas[-1][L]
            abscissas[-1] = x
            weights = np.zeros(len(self.L1_Weights[-1])-1)
            weights[0:-1] = self.L1_Weights[-1][L]
            weights[-1] = w
            
            self.L1_Abscissas.append( abscissas )
            self.L1_Weights.append( weights )
        
        return
        
    def get_Objective_and_Derivative(self, NumberofBasisFunctions=None, scalar=False):
        n = get_Number_of_Polynomials_below_Degree(self.Dimensions, self.MaximumPolynomialDegree) \
              if NumberofBasisFunctions is None else NumberofBasisFunctions
                 
        if scalar: # already square sum
            def Objective_and_Derivative(x):
                num = len(x)//(self.Dimensions+1)
                x = x.reshape((self.Dimensions+1,num)).T
                px = self.PolynomialBasis.evaluate_MonomialBasis(x[:,1:None],NumberofBasisFunctios=n)
                Qp = np.matmul(px,x[:,0])
                f = np.linalg.norm(self.RHS[0:n] - Qp)**2
                
                df = np.zeros_like(x)
                for k in range(num):
                    df[k,0] = -2*np.dot( px[:,k] , self.RHS[0:n] - Qp )
                for i in range(1,self.Dimensions+1):
                    dpxi = self.PolynomialBasis.evaluate_MonomialBasis_Derivative(i-1, x[:,1:None], NumberofBasisFunctios=n)
                    for k in range(num):
                        df[k,i] = -2*np.dot(x[k,0]*dpxi[:,k] , self.RHS[0:n] - Qp)
                df = df.T.reshape(df.size)
                return f,df
        
            self.Objective_and_Derivative = lambda x: Objective_and_Derivative(x)
            
        else: # vector objective and jacobian
            def Objective(x):
                num = len(x)//(self.Dimensions+1)
                x = x.reshape((self.Dimensions+1,num)).T
                px = self.PolynomialBasis.evaluate_MonomialBasis(x[:,1:None],NumberofBasisFunctios=n)
                Qp = np.matmul(px,x[:,0])
                f = (self.RHS[0:n] - Qp) #/ self.RHS[0:n]
                return f
            
            def Objective_Derivative(x):
                num = len(x)//(self.Dimensions+1)
                x = x.reshape((self.Dimensions+1,num)).T
                px = self.PolynomialBasis.evaluate_MonomialBasis(x[:,1:None],NumberofBasisFunctios=n)
                
                df = np.zeros([n,x.size])
                df[:,0:num] = -px #/ self.RHS[0:n][:,None]
                for l in range(1,self.Dimensions+1):
                    dpxl = self.PolynomialBasis.evaluate_MonomialBasis_Derivative(l-1, x[:,1:None], NumberofBasisFunctios=n)
                    df[:,num*l:num*(l+1)] = - x[:,0][None,:]*dpxl #/ self.RHS[0:n][:,None]
                
                return df
        
            self.Objective = lambda x: Objective(x)
            self.Objective_Derivative = lambda x: Objective_Derivative(x)
        
        return
    
    def get_Reduced_Quadrature(self, NumberofBasisFunctions=None,
                               ftol=1e-10, tol=1e-8):
        n = get_Number_of_Polynomials_below_Degree(self.Dimensions, self.MaximumPolynomialDegree) \
              if NumberofBasisFunctions is None else NumberofBasisFunctions
        
        NumberofPoints = n//(self.Dimensions+1) if n % (self.Dimensions+1) == 0 else n//(self.Dimensions+1) + 1

        if len(self.L1_Weights[-1]) > NumberofPoints: # find correct index in Cluster Sequence
            self.get_Cluster_Sequence(min_size=NumberofPoints)
        ind = 0
        while not len(self.L1_Weights[ind]) == NumberofPoints:
            ind += 1

        boolean = True
        while boolean:
            x0 = np.zeros([self.Dimensions+1,NumberofPoints])
            x0[0] = self.L1_Weights[ind]
            x0[1:None] = self.L1_Abscissas[ind].T
            x0 = x0.reshape(x0.size)           
    
            lb = list(np.zeros(x0.size))
            ub = np.ones(x0.size)
            ub[0:NumberofPoints] = np.inf
            ub = list(ub)
            res = scipy.optimize.least_squares(self.Objective,x0,
                                           jac=self.Objective_Derivative, 
                                           bounds = (lb,ub),
                                           ftol=ftol)
            
            print(n,NumberofPoints,res.cost)
            boolean = not (res.success and res.cost < tol)
            ind -= 1
            NumberofPoints += 1
    
        
        x = res.x.reshape([self.Dimensions+1,NumberofPoints-1]).T
        self.Weights.append( x[:,0] )
        self.Abscissas.append( x[:,1:None] )
            
        return res
    
    def get_Reduced_Quadrature_Sequence(self, maxdeg = None, ftol=1e-10, tol=1e-8):
        maxdeg = self.MaximumPolynomialDegree if maxdeg is None else maxdeg
        for i in range(maxdeg,0,-1):
            n_degi = get_Number_of_Polynomials_below_Degree(self.Dimensions,i)
            self.get_Objective_and_Derivative(scalar=False,NumberofBasisFunctions=n_degi)
            self.get_Reduced_Quadrature(NumberofBasisFunctions=n_degi,ftol=ftol,tol=tol)
        return
    
    def get_Reduced_Quadrature_scalar(self, NumberofBasisFunctions=None, 
                                      ftol=1e-12, tol=1e-8):
        n = get_Number_of_Polynomials_below_Degree(self.Dimensions, self.MaximumPolynomialDegree) \
              if NumberofBasisFunctions is None else NumberofBasisFunctions
        
        NumberofPoints = n//(self.Dimensions+1) if n % (self.Dimensions+1) == 0 else n//(self.Dimensions+1) + 1

        if len(self.L1_Weights[-1]) > NumberofPoints: # find correct index in Cluster Sequence
            self.get_Cluster_Sequence(min_size=NumberofPoints)
        ind = 0
        while not len(self.L1_Weights[ind]) == NumberofPoints:
            ind += 1

        boolean = True
        while boolean:
            x0 = np.zeros([self.Dimensions+1,NumberofPoints])
            x0[0] = self.L1_Weights[ind]
            x0[1:None] = self.L1_Abscissas[ind].T
            x0 = x0.reshape(x0.size)
            
            bounds = []
            for i in range(NumberofPoints): bounds.append( (0,None) )
            for i in range(NumberofPoints*self.Dimensions): bounds.append( (0,1) )
            
            res = minimize(self.Objective_and_Derivative, x0, jac=True,
                           method='SLSQP', bounds=bounds,
                           options={'maxiter':1e3,'ftol':ftol})
            
            print(n,NumberofPoints,res.cost)
            boolean = not (res.success and res.fun < tol)
            ind -= 1
            NumberofPoints += 1
                    
        x = res.x.reshape([self.Dimensions+1,NumberofPoints-1]).T
        self.Weights.append( x[:,0] )
        self.Abscissas.append( x[:,1:None] )
            
        return res
    

#%% Functions for ndPolynomial Class

def get_Number_of_Polynomials_below_Degree(d,m):
    return int(sum(scipy.special.binom(g+d-1,d-1) for g in range(m+1)))

def get_zero_positions(d,g):
    # d ... dimension, g ... degree of polynomial
    zeros_positions = np.array(list(itertools.combinations(range(g+d-1), d-1))) 
    return zeros_positions

def get_alphas_for_fixed_degree(d,g):
    # d ... dimension, g ... degree of polynomial 
    zeros_positions = get_zero_positions(d,g)
    
    left_box_wall_positions = -1*(np.ones(len(zeros_positions))).reshape(len(zeros_positions),1)
    right_box_wall_positions = (g+d-1)*(np.ones(len(zeros_positions))).reshape(len(zeros_positions),1)
    zeros_positions_extended = np.concatenate((left_box_wall_positions,zeros_positions,right_box_wall_positions),axis = 1)
    
    alpha = np.diff(zeros_positions_extended,axis = 1) - 1
    return alpha

def get_all_alphas(d,m):
    # d ... dimension, m ... maximum degree of polynomial
    n = int(sum(scipy.special.binom(g+d-1,d-1) for g in range(m+1)))
    alphas = np.zeros([n,d],dtype=int)
    i_old = 0
    for g in range(m+1):
        alphas_fixed_degree = get_alphas_for_fixed_degree(d,g)
        i_new = i_old + len(alphas_fixed_degree)
        alphas[i_old:i_new,:] = alphas_fixed_degree
        i_old = i_new 
    return alphas

def monom_der(i,alphas,p):
    Bi = np.zeros([len(alphas),p.shape[0]])
    ei = np.zeros(alphas.shape[1])
    ei[i] = 1
    for l in range(len(alphas)):
        if alphas[l,i] != 0:
            Bi[l] = alphas[l,i]*np.product((p[:,0:None] ** (alphas[l] - ei)[None,:]),axis = 1)
    return Bi
    
def monom_int(alphas):
    return np.prod(np.mod(alphas+1,2),axis=1) * np.prod(2 / (alphas + 1),axis=1)

def monom_int01(alphas):
    return np.prod(1 / (alphas + 1),axis=1)

def Monom_Int_Gauss(n,a,b):
    # global c
    # c += 1
    if n == 0:
        I = np.sqrt(np.pi) / 2 * (math.erf(b) - math.erf(a))
    elif n == 1:
        I = (np.exp(-a**2) - np.exp(-b**2)) / 2
    else:
        I = (n-1) / 2 * Monom_Int_Gauss(n-2,a,b) + (np.exp(-a**2) * a**(n-1) - np.exp(-b**2) * b**(n-1)) / 2
    return I

def Monom_Int_Gauss_Shifted(n,a,b,ceta,mu):
    return sum(scipy.special.binom(n,k) * Monom_Int_Gauss(k, ceta*(a-mu), ceta*(b-mu)) * mu**(n-k) / (ceta)**(k+1) for k in range(n+1))
       
def Monom_Int_Gauss_multidim(alpha,a,b,ceta,mu):
    ceta = ceta*np.ones(len(mu)) if len(np.array(ceta).shape) == 0 else ceta
    Is = [Monom_Int_Gauss_Shifted(alpha[i], a[i], b[i], ceta[i], mu[i]) for i in range(len(alpha))]
    return np.product(np.array(Is))

def Monom_Int_Gauss_sum(alpha,a,b,A,ceta,mus):
    return sum(A[i] * Monom_Int_Gauss_multidim(alpha, a, b, ceta, mus[i]) for i in range(len(A)))
    
def get_RHS_rbf(alphas,a,b,A,ceta,mus):
    return np.array([Monom_Int_Gauss_sum(alpha,a,b,A,ceta,mus) for alpha in alphas])

def Fast_Monom_Int_Gauss(n,a,b):
    I = np.zeros(n+1)
    I[0] = np.sqrt(np.pi) / 2 * (math.erf(b) - math.erf(a))
    I[1] = (np.exp(-a**2) - np.exp(-b**2)) / 2
    for k in range(2,n+1):
        I[k] = (k-1) / 2 * I[k-2] + \
            (np.exp(-a**2) * a**(k-1) - np.exp(-b**2) * b**(k-1)) / 2
    return I

def Fast_Monom_Int_Gauss_Shifted(n,a,b,ceta,mu):
    I_shift = Fast_Monom_Int_Gauss(n,ceta*(a-mu),ceta*(b-mu))
    I = np.zeros(n+1)
    for k in range(n+1):
        I[k] = sum(scipy.special.binom(k,l) * I_shift[l] * mu**(k-l) / ceta**(l+1) for l in range(k+1))
    return I

def get_unique_Monom_Integrals(maxdeg,a,b,cetas,mus):
    mus = mus.reshape((1,len(mus))) if len(mus.shape) == 1 else mus
    if len(np.array(cetas).shape) == 0:
        cetas = np.ones_like(mus) * cetas
    elif len(np.array(cetas).shape) == 1:
        cetas = np.ones_like(mus) * cetas[None,:]
    maxdeg = maxdeg*np.ones(mus.shape[1],dtype=int) if len(np.array(maxdeg).shape) == 0 else maxdeg
    I = []
    index = []
    for d in range(mus.shape[1]):
        mu_cetas_unique, index_d = np.unique(np.vstack((mus[:,d],cetas[:,d])).T,axis=0,return_inverse=True)
        mu_unique = mu_cetas_unique[:,0]
        cetas_unique = mu_cetas_unique[:,1]
        index.append( index_d )
        I.append( np.array( \
          [Fast_Monom_Int_Gauss_Shifted(maxdeg[d],a[d],b[d],cetas_unique[i],mu_unique[i]) \
            for i in range(len(mu_unique))]) )
    
    return I,np.array(index)

def Fast_Monom_Int_Gauss_Sum(alpha,A,I_unique,index):
    I = 0
    for i in range(len(A)):
        tmp = 1
        for d in range(len(index)):
            tmp *= I_unique[d][index[d,i],alpha[d]]
        I += A[i] * tmp
    return I

def Fast_get_RHS_rbf(alphas,a,b,A,ceta,mus):
    maxdeg = np.max(alphas,axis=0)
    I_unique, index = get_unique_Monom_Integrals(maxdeg,a,b,ceta,mus)
    
    return np.array([Fast_Monom_Int_Gauss_Sum(alpha,A,I_unique,index) for alpha in alphas])
     

#%% Multvariate Polynomials

class ndPolynomial(object):
    
    def __init__(self, Dimensions = 1, maxdeg = 0):
        self.Dimensions = Dimensions
        self.MonomialBasis = get_all_alphas(Dimensions,maxdeg)
        self.MonomIntegrals = None
        self.MaximumDegree = maxdeg
        self.Coefficients = np.zeros(len(self.MonomialBasis))
        self.Degree = 0
        self.Kernel = None
        return
    
    def __copy__(self):
        poly = ndPolynomial(Dimensions=self.Dimensions,maxdeg=self.MaximumDegree)
        poly.Coefficients = np.copy(self.Coefficients)
        poly.MonomialBasis = np.copy(self.MonomialBasis)
        poly.MonomIntegrals = np.copy(self.MonomIntegrals)
        poly.Kernel = self.Kernel
        poly.Degree = self.Degree
        return poly
    
    def __add__(self, other):
        if self.MaximumDegree == other.MaximumDegree:
            poly_sum = self.copy()
            coeff1 = self.Coefficients.copy()
            coeff2 = other.Coefficients.copy()
            
        elif self.MaximumDegree > other.MaximumDegree:
            poly_sum = self.copy()
            coeff1 = self.Coefficients.copy()
            if len(other.Coefficients.shape) == 1:
                coeff2 = np.zeros(len(self.MonomialBasis))
                coeff2[0:len(other.MonomialBasis)] = other.Coefficients.copy()
            else:
                coeff2 = np.zeros([other.Coefficients.shape[0],len(self.MonomialBasis)])
                coeff2[:,0:len(other.MonomialBasis)] = other.Coefficients.copy()
                
        else:
            poly_sum = other.copy()
            coeff1 = other.Coefficients.copy()
            if len(self.Coefficients.shape) == 1:
                coeff2 = np.zeros(len(other.MonomialBasis))
                coeff2[0:len(self.MonomialBasis)] = self.Coefficients.copy()
            else:
                coeff2 = np.zeros([self.Coefficients.shape[0],len(other.MonomialBasis)])
                coeff2[:,0:len(self.MonomialBasis)] = self.Coefficients.copy()

            
        poly_sum.Coefficients = coeff1 + coeff2
        poly_sum.get_Degree()
        return poly_sum

                

        return poly_sum
    
    def __mul__(self, other):
        if (type(other) is float) or (type(other) is int):
            poly_mul = self.copy()
            poly_mul.Coefficients = other*self.Coefficients
        return poly_mul
    
    def __rmul__(self, other):
        if (type(other) is float) or (type(other) is int):
            poly_mul = self.copy()
            poly_mul.Coefficients = other*self.Coefficients
        return poly_mul
    
    def get_Degree(self):
        if len(self.Coefficients.shape) == 1:
            deg = int(np.sum(self.MonomialBasis[self.Coefficients != 0,:][-1]))
        else:
            deg = np.zeros(self.Coefficients.shape[0],dtype=int)
            for i in range(self.Coefficients.shape[0]):
                deg[i] = np.sum(self.MonomialBasis[self.Coefficients[i] != 0,:][-1])
        self.Degree = deg
        return deg
    
    def evaluate(self, points):
        if len(points.shape) != 2: points = points.reshape((1,points.shape[0]))
        coeff = self.Coefficients.reshape((1,len(self.MonomialBasis))) if len(self.Coefficients.shape) !=2 else self.Coefficients.copy()
        result = np.zeros([len(coeff),len(points)])
        for i in range(len(self.MonomialBasis)):
            if np.any(coeff[:,i] != 0):
                alpha = self.MonomialBasis[i]
                result += coeff[:,i][:,None] * np.prod(points ** alpha, axis = 1)[None,:]
                # eval_alpha = np.prod(points ** alpha, axis = 1)
                # for j in range(len(coeff)):
                #     if coeff[j,i] != 0:
                #         result[j] += coeff[j,i] * eval_alpha
        return result
    
    def evaluate_MonomialBasis(self, points, NumberofBasisFunctios = None):
        n = len(self.MonomialBasis) if NumberofBasisFunctios is None else NumberofBasisFunctios
        if len(points.shape) != 2: points = points.reshape((1,points.shape[0]))
        result = np.zeros([n,len(points)])
        for i in range(n):
            alpha = self.MonomialBasis[i]
            result[i] = np.prod(points ** alpha, axis = 1)
        return result
    
    def evaluate_Derivative(self, i, points):
        if len(points.shape) != 2: points = points.reshape((1,points.shape[0]))
        coeff = self.Coefficients.reshape((1,len(self.MonomialBasis))) if len(self.Coefficients.shape) !=2 else self.Coefficients.copy()
        result = np.zeros([len(coeff),len(points)])
        ei = np.zeros(self.Dimensions)
        ei[i] = 1
        for k in range(len(self.MonomialBasis)):
            if np.any(coeff[:,k] != 0):
                alpha = self.MonomialBasis[k]
                if alpha[i] != 0:
                    result += coeff[:,k][:,None] * (alpha[i] * np.prod(points ** (alpha-ei), axis = 1))[None,:]
        return result
    
    def evaluate_MonomialBasis_Derivative(self, i, points, NumberofBasisFunctios = None):
        n = len(self.MonomialBasis) if NumberofBasisFunctios is None else NumberofBasisFunctios
        if len(points.shape) != 2: points = points.reshape((1,points.shape[0]))
        result = np.zeros([n,len(points)])
        ei = np.zeros(self.Dimensions)
        ei[i] = 1
        for k in range(n):
            alpha = self.MonomialBasis[k]
            if alpha[i] != 0:
                result[k] = alpha[i] * np.prod(points ** (alpha-ei), axis = 1)
        return result

    def get_MonomIntegrals(self, analytic = False, N_points_1d = 10):
        if self.Kernel is None:
            if analytic:
                self.MonomIntegrals = monom_int01(self.MonomialBasis)
            else:
                self.MonomIntegrals = np.zeros(len(self.MonomialBasis))
                TQF = TensorProduct_Quadrature(Dimensions=self.Dimensions)
                TQF.set_UnivariateBasis(types=['l'],NumberofPoints=[N_points_1d])
                TQF.get_Tensor_Quadrature()
                for i in range(len(self.MonomialBasis)):
                    alpha = self.MonomialBasis[i]
                    f = np.prod(TQF.Abscissas ** alpha, axis=1)
                    self.MonomIntegrals[i] = TQF.Integrate(f)
                                        
        elif callable(self.Kernel):
            if not analytic:
                self.MonomIntegrals = np.zeros(len(self.MonomialBasis))
                TQF = TensorProduct_Quadrature(Dimensions=self.Dimensions)
                TQF.set_UnivariateBasis(types=['l'],NumberofPoints=[N_points_1d])
                TQF.get_Tensor_Quadrature()
                for i in range(len(self.MonomialBasis)):
                    alpha = self.MonomialBasis[i]
                    f = np.prod(TQF.Abscissas ** alpha, axis=1) * self.Kernel(TQF.Abscissas)
                    self.MonomIntegrals[i] = TQF.Integrate(f)
                    
        elif type(self.Kernel) is list:
            if analytic:
                alphas = self.MonomialBasis
                A = self.Kernel[0]
                ceta = self.Kernel[1]
                mus = self.Kernel[2]
                a = np.zeros(self.Dimensions)
                b = np.ones(self.Dimensions)
                self.MonomIntegrals = Fast_get_RHS_rbf(alphas, a, b, A, ceta, mus)
            else:
                self.MonomIntegrals = np.zeros(len(self.MonomialBasis))
                TQF = TensorProduct_Quadrature(Dimensions=self.Dimensions)
                TQF.set_UnivariateBasis(types=['l'],NumberofPoints=[N_points_1d])
                TQF.get_Tensor_Quadrature()
                for i in range(len(self.MonomialBasis)):
                    alpha = self.MonomialBasis[i]
                    f = np.prod(TQF.Abscissas ** alpha, axis=1) * eval_rbf_Kernel(TQF.Abscissas,*self.Kernel)
                    self.MonomIntegrals[i] = TQF.Integrate(f)
        return
    
    def get_Integrals(self):
        return np.matmul(self.Coefficients,self.MonomIntegrals)
            
    
#%% functions for ndBasis class

def get_combinations(maxdeg_vec,trig_number):
    vec_list = [np.arange(maxdeg_vec[i]+1) for i in range(len(maxdeg_vec))]
    vec_list = vec_list + trig_number*[np.arange(2)]
    combinations = np.array(list(itertools.product(*vec_list)))
    return combinations

def SymOp(sym_op,x):
    return (sym_op[0]@x.T + sym_op[1]).T

def rescale_trig_dims(points, ranges, trig_indices):
    rescaled_points = points.copy()
    if len(trig_indices) > 0:
        rescaled_points[:,trig_indices] = points[:,trig_indices] * (np.diff(ranges, axis = 1).T) + ranges[:,0][None,:]
    return rescaled_points

        
#%% Basis Class - polynomial and trigonometric

class ndBasis(object):
    
    def __init__(self, dim = 1, maxdeg = 0, maxdeg_factor_vec = None, basis_type = None, 
                 sym_ops = [], periodicity_factor = None, ranges = None):
        self.basis_type = basis_type if basis_type is not None else np.array(dim*["p"])
        self.trig_indices = np.where(self.basis_type == "t")[0]
        self.pol_indices = np.where(self.basis_type == "p")[0]
        self.trig_number = len(self.trig_indices)
        self.dim = dim
        
        self.maxdeg = maxdeg
        self.maxdeg_factor_vec = maxdeg_factor_vec if maxdeg_factor_vec is not None else np.ones(self.dim)
        self.maxdeg_vec = np.floor(self.maxdeg / self.maxdeg_factor_vec)
        
        self.Kernel = None
        self.sym_ops = sym_ops
        self.periodicity_factor = np.ones_like(self.trig_indices) if periodicity_factor is None else periodicity_factor
        
        self.ranges = ranges if ranges is not None else np.array(self.trig_number*[[0,2*np.pi]])
        
        self.Basis,_,_ = self.get_alphas()
        self.basis_indices_below_degree = [np.where(np.sum(self.Basis[:,0:self.dim]*self.maxdeg_factor_vec,axis = 1) <= i)[0] for i in range(maxdeg+1)]
        self.BasisIntegrals = None
        
        return
    
    def eval_basis_fun(self,x,alpha,rescale=True):
        x = x if len(x.shape) > 1 else x.reshape(1,len(x))
        x = rescale_trig_dims(x, self.ranges, self.trig_indices) if rescale else x
       
        val = np.prod(np.sin(alpha[self.trig_indices]*self.periodicity_factor*x[:,self.trig_indices] + np.pi/2 * alpha[self.dim:None]),axis=1) * \
            np.prod(x[:,self.pol_indices] ** alpha[self.pol_indices],axis=1)
            
        return val
        
    
    def get_alphas(self):
        alphas = get_combinations(self.maxdeg_vec, self.trig_number)
        L = (np.sum(alphas[:,0:self.dim]*self.maxdeg_factor_vec,axis = 1) <= self.maxdeg) * \
            np.all(alphas[:,self.trig_indices] + alphas[:,self.dim:None] >= 1,axis=1)
        alphas = alphas[L]
        
        sig_list = []
        sieved_alphas = []
        for alpha in alphas:
            L,sig = self.get_signum(alpha)
            sig_list.append(sig)
            if L:
                sieved_alphas.append(alpha)
        sieved_alphas = np.array(sieved_alphas)
        sort_indices = np.argsort(np.sum(sieved_alphas[:,0:self.dim],axis=1))
        sieved_alphas = sieved_alphas[sort_indices]
        return sieved_alphas, alphas, np.array(sig_list)
    
    
    def get_signum(self,alpha):
        if len(self.sym_ops) == 0:
            return True,None
        else:
            x = np.random.rand(self.dim)
            x_trans = np.array([SymOp(self.sym_ops[i],x) for i in range(len(self.sym_ops))])
            sig = self.eval_basis_fun(x_trans, alpha,rescale=False) / self.eval_basis_fun(x, alpha,rescale=False)
            #print(self.eval_basis_fun(x_trans, alpha) , self.eval_basis_fun(x, alpha))
        return not np.any(np.abs(sig + 1) < 1e-3), sig
       
    
    def eval_basis_funs(self,x,maxdeg=None,rescale=True):
        maxdeg = self.maxdeg if maxdeg is None else maxdeg
        x = x if len(x.shape) > 1 else x.reshape(1,len(x))
        x = rescale_trig_dims(x, self.ranges, self.trig_indices) if rescale else x
        n = len(self.basis_indices_below_degree[maxdeg])
        result = np.zeros([n,len(x)])
        for i in range(n):
            result[i] = self.eval_basis_fun(x, self.Basis[self.basis_indices_below_degree[maxdeg]][i],rescale=False)
        
        return np.array(result)
    
    def get_BasisIntegrals(self, analytic = False, N_points_1d = 10):
        if self.Kernel is None:
            if analytic:
                print("not possible yet")
            else:
                TQF = TensorProduct_Quadrature(Dimensions=self.dim)
                TQF.set_UnivariateBasis(types=['l'],NumberofPoints=[N_points_1d])
                TQF.get_Tensor_Quadrature()
                F = self.eval_basis_funs(TQF.Abscissas[0])
                self.BasisIntegrals = F@TQF.Weights[0]
                                        
        elif callable(self.Kernel):
            if not analytic:
                TQF = TensorProduct_Quadrature(Dimensions=self.Dimensions)
                TQF.set_UnivariateBasis(types=['l'],NumberofPoints=[N_points_1d])
                TQF.get_Tensor_Quadrature()
                F = self.eval_basis_funs(TQF.Abscissas[0]) * self.Kernel(TQF.Abscissas[0])[None,:]
                self.BasisIntegrals = F@TQF.Weights[0]
                
                    
        return self.BasisIntegrals
    
    
#%% test ndBasis

# A1 = np.diag([-1,1,-1])
# b1 = np.array([np.pi,0,np.pi])
# s1 = (A1,b1)

# A2 = np.diag([-1,1,1])
# b2 = np.array([0,np.pi,np.pi])
# s2 = (A2,b2)


# B = ndBasis(dim = 3, maxdeg = 8, maxdeg_factor_vec=np.array([1,1,1]),basis_type=np.array(["t","t","t"]),
#             sym_ops = [s1,s2])
    
# alp,alp_old,sig_list = B.get_alphas()


#%% tests

# T = TensorProduct_Quadrature(Dimensions = 2)
# T.set_UnivariateBasis(types=['l'],NumberofPoints=[5])
# T.get_Tensor_Quadrature()

# MC = MonteCarlo_Quadrature(Dimensions = 2)
# MC.get_MC_Abscissas_and_Weights(NumberofPoints=100)

# SQ = Smolyak_Quadrature(Dimensions=2)
# SQ.set_UnivariateQuadratureSequence(typ=['c'],NumberofPoints=[1,3,5,9,17,33])
# SQ.get_Smolyak_Quadrature(Level=2,nested=True)

# x = SQ.Abscissas[:,0]
# y = SQ.Abscissas[:,1]

# plt.scatter(x,y)
# plt.axis('square')
# plt.xlim(-0.1,1.1)
# plt.ylim(-0.1,1.1)

#%% check interpolation

# x = np.random.rand(500,2)
# fun = lambda x: x[:,0]**2 #np.sum(x**2,axis=1)
# z = fun(x)

# rbfs = Rbf(x[:,0],x[:,1],z,function = "gaussian",epsilon = 0.3)

#%% check polynomials

# p = ndPolynomial(Dimensions = 2, maxdeg = 3)
# p.Coefficients = np.zeros([2,len(p.MonomialBasis)])
# p.Coefficients[0,1] = 1
# p.Coefficients[0,0] = 1
# p.Coefficients[1,2] = 1
# q = ndPolynomial(Dimensions = 2, maxdeg = 2)
# q.Coefficients[2] = 5
# s = ndPolynomial(Dimensions = 2, maxdeg = 3)
# s.Coefficients = np.zeros([2,len(s.MonomialBasis)])
# s.Coefficients[0,5] = 9
# s.Coefficients[1,7] = 3

# points = np.random.rand(5,2)
# r = p.evaluate_MonomialBasis(points)
# r = p.evaluate_MonomialBasis_Derivative(0, points)

# s.Kernel = lambda x: x[:,0]**2
# s.get_MonomIntegrals(analytic=False)

#%% test L1 Quadrature 2

# L1 = L1_Quadrature_2(Dimensions=4)
# L1.set_Basis(maxdeg=3,basis_type=np.array(["t","t","t","t"]))
# L1.get_Random_AbscissasProposal(2500)
# L1.get_RHS()
# L1.get_L1_Quadrature(Degree=3)
# L1.get_nested_L1_Quadrature_Sequence()

#%% test L1D Quadrature
# dim = 4
# maxdeg = 5
# analytic = False

# Kernel = None
# Kernel = lambda x: x[:,0]**4 * np.exp(np.sin(np.sum(x,axis=1))**2)

# L1D = L1D_Quadrature(Dimensions=dim)
# L1D.Kernel = Kernel
# L1D.set_PolynomialBasis(MaximumDegree=maxdeg)
# L1D.get_RHS(analytic = analytic)
# n = int( get_Number_of_Polynomials_below_Degree(dim,maxdeg)) * 4
# L1D.get_Random_AbscissasProposal(NumberofPoints=n)

# L1D.get_nested_L1D_Quadrature_Sequence()

# print([get_Number_of_Polynomials_below_Degree(dim, m) for m in range(maxdeg+1)])
# print([len(L1D.Weights[i]) for i in range(maxdeg,-1,-1)])
#%% test Reduced Quadrature
# dim = 4
# maxdeg = 4
# RQ = Reduced_Quadrature(Dimensions=dim)
# RQ.set_PolynomialBasis(MaximumDegree=maxdeg)
# RQ.get_RHS()
# n = get_Number_of_Polynomials_below_Degree(dim,maxdeg)
# RQ.get_Random_AbscissasProposal(NumberofPoints=4*n)
# RQ.get_L1_Abscissas_and_Weights()
# RQ.get_Cluster_Sequence()

# n = get_Number_of_Polynomials_below_Degree(dim, 4)
# RQ.get_Objective_and_Derivative(scalar=True,NumberofBasisFunctions=n)
# RQ.get_Objective_and_Derivative(scalar=False,NumberofBasisFunctions=n)


# tic()
# res = RQ.get_Reduced_Quadrature(NumberofBasisFunctions=n)
# toc()

# p = ndPolynomial(Dimensions=dim,maxdeg=maxdeg)
# p.Coefficients[[0,1,4,6,7,9,14]] = 1
# p.get_MonomIntegrals(analytic=True)
# I_a = p.get_Integrals()
# f = p.evaluate(RQ.Abscissas[0])
# f = f.reshape(f.size)

# I_n = RQ.Integrate(f)

# print(I_a,I_n)

#%% test Fast Gauss Int

# dim = 3
# N = 100

# A = np.random.rand(N)
# mus = np.random.rand(N,dim)
# ceta = np.array([1,0.8,0.7])

# alphas = get_all_alphas(dim,5)
# a = np.zeros(dim)
# b = np.ones(dim)

# tic()
# rhs = get_RHS_rbf(alphas, a, b, A, ceta, mus)
# toc()

# tic()
# rhsf = Fast_get_RHS_rbf(alphas,a,b,A,ceta,mus)
# toc()
     























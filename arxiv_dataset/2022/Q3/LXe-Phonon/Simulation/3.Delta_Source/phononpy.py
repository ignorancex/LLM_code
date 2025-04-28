#################################################################
#                                                               #
#       PhononPy a library for sound physics                    #
#                                                               #
# Phononpy is a project to calculate the acoustic signals       #
# that particles create in various noninteracting liquids       #
#                                                               #
# It uses multiprocessing to run on a machine that supports     #
# parallel computing.                                           #
#                                                               #
# Maintained by: Panos Oikonomou (po524@nyu.edu)                #
#                                                               #
#################################################################

import numpy as np
import sympy as sp
from multiprocessing import Pool
from sympy.utilities.lambdify import lambdify
from scipy.special import binom
from tqdm.notebook import tqdm
try:
    from numba import jit,cuda
    import cupy as cp
except ImportError:
    print("CUDA GPU Acceleration is unavailable for your system : (")
    pass

# Coloring output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# We will create a class for the pressure estimator
# It is going to be a list of functions with vectorized inputs that is then calculated
class estimator(object):
    # Constructor
    def __init__(self, max_order:int=0,n_cores:int=2,slow:bool=True,simplify:bool=False,GPU:bool=False):
        self.max_order  = max_order     # Maximum order to calculate for
        self.n_cores    = n_cores       # Number of corse to be used in multiprocessing
        self.GPU        = GPU           # Do you want to use CUDA for evaluating the function?
        self.slow       = slow          # Store if this is for particles with v<1
        self.terms      = []            # A list for the terms to be calculated

        # Create the terms
        print(bcolors.BOLD+'Generating Estimator for order %d'%self.max_order+bcolors.ENDC)
        self.assemble_multiprocessing(simplify=simplify)
        if self.GPU: print(bcolors.BOLD+bcolors.WARNING+'WARNING!'+bcolors.ENDC+' Since you are using GPU accelleration you need to check that your input satisfies these conditions z < v t and (z-v t)^2 > r^2 (v^2 - 1). Otherwise there will be errors')
    
    # Coefficient
    def C(self,n:int,m:int):
        return 2**m /(4**(n+1) + np.pi * np.math.factorial(m)) * binom(2*n - m,n)

    # Define the source function
    def q(self,t):
        return 1/(1+sp.exp(-5 * (t-2)))

    # Solutions
    def T(self,r,z,t,v,sgn=1):
        gamma_sq = 1/(1-v**2)
        return gamma_sq*(t- v*z) + sgn*sp.sqrt(gamma_sq**2*(z-v*t)**2 + gamma_sq*r**2)

    # Get slow term
    def get_slow_term(self,n:int,m:int,
                        r:sp.Symbol=sp.Symbol('r'),
                        z:sp.Symbol=sp.Symbol('z'),
                        t:sp.Symbol=sp.Symbol('t'),
                        v:sp.Symbol=sp.Symbol('v')):
        # Variable
        T = sp.Symbol('T')

        # Calculate the derivative
        der = sp.diff((t-T)**m * self.q(T)/(v*(z-v*T) - sp.sqrt(r**2 + (z-v*T)**2)),T,m)
        
        # Substitute solutions
        der = der.subs(T,self.T(r,z,t,v,sgn=-1))

        # Calculate the final derivative
        return sp.diff(der,t,n+1)

    # Get fast term
    def get_fast_term(self,n:int,m:int,
                        r:sp.Symbol=sp.Symbol('r'),
                        z:sp.Symbol=sp.Symbol('z'),
                        t:sp.Symbol=sp.Symbol('t'),
                        v:sp.Symbol=sp.Symbol('v')):
        # Variable
        T = sp.Symbol('T')

        # Calculate the derivative
        der = sp.diff((t-T)**m * self.q(T)/(v*(z-v*T) - sp.sqrt(r**2 + (z-v*T)**2)),T,m)
        
        # Substitute solutions
        der = der.subs(T,self.T(r,z,t,v,sgn=-1)) + der.subs(T,self.T(r,z,t,v,sgn=1))

        # Calculate the final derivative
        return sp.diff(der,t,n+1)

    # Get a pickled term
    def get_pickled_slow_term(self,n:int,m:int,\
                        simplify:bool=False,
                        r:sp.Symbol=sp.Symbol('r'),
                        z:sp.Symbol=sp.Symbol('z'),
                        t:sp.Symbol=sp.Symbol('t'),
                        v:sp.Symbol=sp.Symbol('v'),
                        L:sp.Symbol=sp.Symbol('L'),
                        append:bool=False):
        if not append: 
            f = L**n * self.C(n,m) * self.get_slow_term(n,m,r,z,t,v)
            if simplify: f = sp.simplify(f)
            print("\t"+bcolors.OKCYAN+"Slow:"+bcolors.ENDC+"%2d,%2d "%(n,m)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
            return f
        else:
            f = L**n * self.C(n,m) * self.get_slow_term(n,m,r,z,t,v) 
            if simplify: f = sp.simplify(f)
            self.terms.append(lambdify([r,z,t,v,L],f,'numpy'))
            print("\t"+bcolors.OKCYAN+"Slow:"+bcolors.ENDC+"%2d,%2d "%(n,m)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
        
    def get_pickeld_fast_term(self,n:int,m:int,\
                        simplify:bool=False,
                        r:sp.Symbol=sp.Symbol('r'),
                        z:sp.Symbol=sp.Symbol('z'),
                        t:sp.Symbol=sp.Symbol('t'),
                        v:sp.Symbol=sp.Symbol('v'),
                        L:sp.Symbol=sp.Symbol('L'),
                        append:bool=False):
        
        if not append: 
            f = L**n * self.C(n,m) * self.get_fast_term(n,m,r,z,t,v)
            if simplify: f = sp.simplify(f)
            print("\t"+bcolors.OKBLUE+"Fast:"+bcolors.ENDC+"%2d,%2d "%(n,m)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
            return f
        
        else:
            f = lambdify([r,z,t,v,L], L**n * self.C(n,m) * self.get_fast_term(n,m,r,z,t,v),'numpy')
            if simplify: f = sp.simplify(f)
            g = (lambda F: lambda r,z,t,v,L: 0 if abs(z-v*t) <= r*(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,L))(f)
            self.terms.append(np.vectorize(g,excluded=['v','L']))
            print("\t"+bcolors.OKBLUE+"Fast:"+bcolors.ENDC+"%2d,%2d "%(n,m)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)


    # If you want to use multiprocessing
    # This function will assemble all the proccesses needed
    def get_permutations(self,simplify:bool=False,variables:bool=False):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')
        
        # Create a list of permutations
        permutations = []
        for n in range(self.max_order+1):
            for m in range(n+1):
                    entry = [n,m]
                    entry.append(simplify)
                    if variables: entry += [r,z,t,v,L]
                    permutations.append(entry)
                    
        return permutations

    # Initialize using multiprocessing pool
    def assemble_multiprocessing(self,simplify:bool=False):
        print("Assembling the estimator using multiprocessing on %d cores"%self.n_cores)
        
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')

        # Get the arguments
        permutations = self.get_permutations(simplify=simplify)
        print("Permuations created successfully")
        
        # Get a pool
        pool = Pool(self.n_cores)
        
        # Start the pool
        if self.slow:
            print('\tSend the slow data to the pool')
            terms = pool.starmap_async(self.get_pickled_slow_term,permutations)
        
        else:
            print('\tSend the fast data to the pool')
            terms = pool.starmap_async(self.get_pickeld_fast_term,permutations)
        
        # Close the pool
        pool.close()
        pool.join()
        print('Pool closed Successfully')

        # Assign the functions to the appropriate arrays
        print('recasting functions')
        TERMS = terms.get()
        for f in tqdm(TERMS):
            if self.slow:
                if self.GPU: self.terms.append(lambdify([r,z,t,v,L],f,'cupy'))
                else: self.terms.append(lambdify([r,z,t,v,L],f,'numpy'))
            else:
                if self.GPU: self.terms.append(lambdify([r,z,t,v,L],f,'cupy'))
#                     f = lambdify([r,z,t,v,L], sp.Piecewise( (0,z>=v*t),(0,(z-v*t)**2 <= r**2*(v**2 - 1)),(f,True)),'cupy')
#                     g = (lambda F: lambda r,z,t,v,L: 0 if abs(z-v*t) <= r*np.sqrt(v**2 - 1) or (z>v*t) else F(r,z,t,v,L))(f)
#                     self.terms.append(f)
                    
                else:
                    f = lambdify([r,z,t,v,L], f,'numpy')
                    g = (lambda F: lambda r,z,t,v,L: 0 if abs(z-v*t) <= r*np.sqrt(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,L))(f)
                    self.terms.append(np.vectorize(g,excluded=['v','L']))

        print(bcolors.BOLD+bcolors.UNDERLINE+'Estimator Generated Successfully'+bcolors.ENDC)
        
        


    # Actually make a prediction given a set of data 
    def __call__(self,r,z,t,v,l):
        if v == 1: v = 0.9999999
        sum = 0
        for f in self.terms: sum += f(r,z,t,v,l)
        return sum
    
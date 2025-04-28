# include necessary libraries
import numpy as np
import sympy as sp
from multiprocessing import Pool
from sympy.utilities.lambdify import lambdify
from scipy.special import binom
from tqdm.notebook import tqdm

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
class estimator:
    # Constructor
    def __init__(self,max_order:int=0, multiprocess=False, n_cores:int=2, simplify:bool=False):
        self.max_order  = max_order     # Maximum order to calculate for
        self.n_cores    = n_cores       # Number of corse to be used in multiprocessing
        self.terms_slow = []            # A list for the terms to be calculated for v<1
        self.terms_fast = []            # A list for the terms to be calculated for v>1

        # Now calculate the terms
        if not multiprocess:
            print(bcolors.BOLD+'Generating Estimator for order %d'%self.max_order+bcolors.ENDC)
            self.assemble_slow()
            self.assemble_fast()
            
        else:
            print(bcolors.BOLD+'Generating Estimator for order %d'%self.max_order+bcolors.ENDC)
            self.assemble_multiprocessing(simplify=simplify)

    # Coefficient
    def C(self, n:int, m:int, l:int):
        return (-2)**m * (-1)**l * np.math.factorial(l) * binom(2*n - m, n) * binom(m,l)**2/ (4**(n+1) * np.pi * np.math.factorial(m))
    
    # Get the slow terms
    def get_slow_term(self,n:int,m:int,l:int):
        # some symbols
        v       = sp.Symbol('v')
        g_sq    = 1/(1-v**2)
        rho     = sp.Symbol('r')
        t       = sp.Symbol('t')
        Z       = sp.Symbol('Z')
        z       = sp.Symbol('z')
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')
        q       = 1/(1+sp.exp(-a*(t-(rho**2 + Z**2)**(1/2)-s)))

        # Calculate the derivatives
        der = v**(m-l)*sp.diff(q*(rho**2 + Z**2)**((m-l)/2)/((rho**2 + Z**2)**(1/2)-v*Z),Z,m-l)

        # Calculate the solution for slow speed
        Z_slow = g_sq * (z - v* t) + v * ((g_sq * (z - v* t))**2 + g_sq * rho**2)**(1/2)

        # Now plug in the solution for slow speed
        der = der.subs(Z,Z_slow)

        # Calculate the time derivative
        der = sp.diff(der,t,n+1)

        # Return the derivative
        return der
        
    # Get the fast terms
    def get_fast_term(self,n:int,m:int,l:int):
        # some symbols
        v       = sp.Symbol('v')
        g_sq    = 1/(1-v**2)
        rho     = sp.Symbol('r')
        t       = sp.Symbol('t')
        Z       = sp.Symbol('Z')
        z       = sp.Symbol('z')
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')
        q       = 1/(1+sp.exp(-a*(t-(rho**2 + Z**2)**(1/2)-s)))

        # Calculate the derivatives
        der = v**(m-l)*sp.diff(q*(rho**2 + Z**2)**((m-l)/2)/((rho**2 + Z**2)**(1/2)-v*Z),Z,m-l)

        # Calculate the solution for +-
        Z_p = g_sq * (z - v* t) + v * ((g_sq * (z - v* t))**2 + g_sq * rho**2)**(1/2)
        Z_m = g_sq * (z - v* t) - v * ((g_sq * (z - v* t))**2 + g_sq * rho**2)**(1/2)

        # Now plug in the solution for slow speed
        der = der.subs(Z,Z_p) + der.subs(Z,Z_m)

        # Calculate the time derivative
        der = sp.diff(der,t,n+1)

        # Return the derivative
        return der

    # Some wrappers to be used in multiprocessing
    def get_slow_term_func(self,n:int,m:int,l:int,\
                           simplify:bool=False,
                           r:sp.Symbol=sp.Symbol('r'),
                           z:sp.Symbol=sp.Symbol('z'),
                           t:sp.Symbol=sp.Symbol('t'),
                           v:sp.Symbol=sp.Symbol('v'),
                           L:sp.Symbol=sp.Symbol('L'),
                           a:sp.Symbol=sp.Symbol('a'),
                           s:sp.Symbol=sp.Symbol('s'),
                           append:bool=False):
        if not append: 
            f = L**n * self.C(n,m,l) * self.get_slow_term(n,m,l)
            if simplify: f = sp.simplify(f)
            print("\t"+bcolors.OKCYAN+"Slow:"+bcolors.ENDC+"%2d,%2d,%2d "%(n,m,l)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
            return f
        else:
            f = L**n * self.C(n,m,l) * self.get_slow_term(n,m,l) 
            if simplify: f = sp.simplify(f)
            self.terms_slow.append(lambdify([r,z,t,v,L,a,s],f,'numpy'))
            print("\t"+bcolors.OKCYAN+"Slow:"+bcolors.ENDC+"%2d,%2d,%2d "%(n,m,l)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
    
    # Some wrappers to be used in multiprocessing
    def get_fast_term_func(self,n:int,m:int,l:int,\
                           simplify:bool=False,
                           r:sp.Symbol=sp.Symbol('r'),
                           z:sp.Symbol=sp.Symbol('z'),
                           t:sp.Symbol=sp.Symbol('t'),
                           v:sp.Symbol=sp.Symbol('v'),
                           L:sp.Symbol=sp.Symbol('L'),
                           a:sp.Symbol=sp.Symbol('a'),
                           s:sp.Symbol=sp.Symbol('s'),
                           append:bool=False):
        
        if not append: 
            f = L**n * self.C(n,m,l) * self.get_fast_term(n,m,l)
            if simplify: f = sp.simplify(f)
            print("\t"+bcolors.OKBLUE+"Fast:"+bcolors.ENDC+"%2d,%2d,%2d "%(n,m,l)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
            return f

        else:
            f = lambdify([r,z,t,v,L,a,s], L**n * self.C(n,m,l) * self.get_fast_term(n,m,l),'numpy')
            if simplify: f = sp.simplify(f)
            g = (lambda F: lambda r,z,t,v,L,a,s: 0 if abs(z-v*t) <= r*(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,L,a,s))(f)
            self.terms_fast.append(np.vectorize(g,excluded=['v','l','a','s']))
            print("\t"+bcolors.OKBLUE+"Fast:"+bcolors.ENDC+"%2d,%2d,%2d "%(n,m,l)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)
    
    # If you want to use multiprocessing
    # This function will assemble all the proccesses needed
    def get_permutations(self,simplify:bool=False,variables:bool=False):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')
        
        # Create a list of permutations
        permutations = []
        for n in range(self.max_order+1):
            for m in range(n+1):
                for l in range(m+1):
                    entry = [n,m,l]
                    entry.append(simplify)
                    if variables: entry += [r,z,t,v,L,a,s]
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
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')

        # Get the arguments
        permutations = self.get_permutations(simplify=simplify)
        print("Permuations created successfully")
        
        # Get a pool
        pool = Pool(self.n_cores)
        
        # Start the pool for the negatives
        print('\tSend the slow data to the pool')
        terms_slow = pool.starmap_async(self.get_slow_term_func,permutations)
        
        print('\tSend the fast data to the pool')
        terms_fast = pool.starmap_async(self.get_fast_term_func,permutations)
        
        # Close the pool
        pool.close()
        pool.join()
        print('Pool closed Successfully')

        # Assign the functions to the appropriate arrays
        print('recasting functions')
        for f in tqdm(terms_slow.get()): 
            self.terms_slow.append(lambdify([r,z,t,v,L,a,s],f,'numpy'))
        for f in tqdm(terms_fast.get()):
            f = lambdify([r,z,t,v,L,a,s], f,'numpy')
            g = (lambda F: lambda r,z,t,v,L,a,s: 0 if abs(z-v*t) <= r*(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,L,a,s))(f)
            self.terms_fast.append(np.vectorize(g,excluded=['v','l','a','s']))

        print(bcolors.BOLD+bcolors.UNDERLINE+'Estimator Generated Successfully'+bcolors.ENDC)
    
    
    # Assembles the terms for the slow particles
    def assemble_slow(self):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')

        # Add the terms
        for n in tqdm(range(self.max_order+1),desc='Assembling Estimator for v<1'):
            for m in tqdm(range(n+1),leave=False):
                for l in tqdm(range(m+1),leave=False):
                    f = lambdify([r,z,t,v,L,a,s],L**n * self.C(n,m,l) * self.get_slow_term(n,m,l),'numpy')
                    self.terms_slow.append(f)
    
    # Assembles the terms for the slow particles
    def assemble_fast(self):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')
        a       = sp.Symbol('a')
        s       = sp.Symbol('s')

        # Add the terms
        for n in tqdm(range(self.max_order+1),desc='Assembling Estimator for v>1'):
            for m in tqdm(range(n+1),leave=False):
                for l in tqdm(range(m+1),leave=False):
                    f = lambdify([r,z,t,v,L,a,s], L**n * self.C(n,m,l) * self.get_fast_term(n,m,l),'numpy')

                    g = (lambda F: lambda r,z,t,v,l,a,s: 0 if abs(z-v*t) <= r*(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,l,a,s))(f)
                    g = np.vectorize(g,excluded=['v','l','a','s'])
                    self.terms_fast.append(g)
    
    # Actually make a prediction given a set of data
    def __call__(self,r,z,t,v,l,a,s):
        if v == 1: v = 0.9999999
        if v < 1:
            sum = 0
            for f in self.terms_slow: sum += f(r,z,t,v,l,a,s)
            return sum
        else:
            sum = 0
            for g in self.terms_fast: sum += g(r,z,t,v,l,a,s)
            return sum
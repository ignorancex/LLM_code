#################################################################
#                                                               #
#       PyPhonon a library for sound physics                    #
#                                                               #
# Pyphonon is a project to calculate the acoustic signals       #
# that particles create in various noninteracting liquids       #
#                                                               #
# It uses multiprocessing to run on a machine that supports     #
# parallel computing.                                           #
#                                                               #
# Maintained by: Panos Oikonomou (po524@nyu.edu)                #
#                                                               #
#################################################################

from numpy import math
import numpy as np
import sympy as sp
import scipy.constants as c
from multiprocessing import Pool
from sympy.utilities.lambdify import lambdify
from scipy.special import binom
from tqdm.notebook import tqdm
try:
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
        self.sterms     = []            # List with the sympy terms of the estimator

        # Create the terms
        print(bcolors.BOLD+'Generating Estimator for order %d'%self.max_order+bcolors.ENDC)
        self.assemble_multiprocessing(simplify=simplify)
        if self.GPU: print(bcolors.BOLD+bcolors.WARNING+'WARNING!'+bcolors.ENDC+' Since you are using GPU accelleration you need to check that your input satisfies v*t - (z+r*np.sqrt(v**2-1)). Otherwise there will be errors')
    
    # Coefficient
    def C(self,n:int,m:int):
        return 2**m /(4**(n+1) + np.pi * math.factorial(m)) * binom(2*n - m,n)

    # Define the source function
    def q(self,t):
        # return sp.Heaviside(t)
        return 1/(1+sp.exp(-1e6 * (t-0)))

    # Solutions
    def T(self,r,z,t,v,sgn=1):
        gamma_sq = 1/(1-v**2)
        # return gamma_sq*(t - v*z) + sgn*sp.sqrt(gamma_sq**2*(z-v*t)**2 + gamma_sq*r**2)
        return gamma_sq*(t - v*z) + gamma_sq*sgn*sp.sqrt((z-v*t)**2 + r**2/gamma_sq)

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

        # print(sp.mathematica_code(der))

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
            g = (lambda F: lambda r,z,t,v,L: 0. if abs(z-v*t) <= r*(v**2 - 1) or (z-v*t) > 0. else F(r,z,t,v,L))(f)
            self.terms.append(np.vectorize(g,excluded=['v','L']))
            print("\t"+bcolors.OKBLUE+"Fast:"+bcolors.ENDC+"%2d,%2d "%(n,m)+bcolors.OKGREEN+"Done!"+bcolors.ENDC)

    # Get Jacobian
    def get_jacobian(self):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')

        # Sum all the terms together
        SUM = sum(self.sterms)

        # Calculate jacobian
        M = sp.Matrix([SUM])
        J = M.jacobian([t,z])

        # Return the lambdified function
        return lambdify([r,z,t,v,L],J,'numpy')

    # Get Hessian
    def get_hessian(self):
        # Define some symbols
        v       = sp.Symbol('v')
        L       = sp.Symbol('L')
        r       = sp.Symbol('r')
        t       = sp.Symbol('t')
        z       = sp.Symbol('z')

        # Sum all the terms together
        SUM = sum(self.sterms)

        # Calculate jacobian
        H = sp.hessian(SUM,[t,z])

        # Return the lambdified function
        return lambdify([r,z,t,v,L],H,'numpy')

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
        self.sterms = TERMS.copy()
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
                    g = (lambda F: lambda r,z,t,v,L: 0. if v*t - (z+r*np.sqrt(v**2-1)) < 0. else F(r,z,t,v,L))(f)
                    # g = (lambda F: lambda r,z,t,v,L: F(r,z,t,v,L))(f)
                    self.terms.append(np.vectorize(g,excluded=['v','L']))

        print(bcolors.BOLD+bcolors.UNDERLINE+'Estimator Generated Successfully'+bcolors.ENDC)
        
        


    # Actually make a prediction given a set of data 
    def __call__(self,r,z,t,v,l):
        if v == 1: v = 0.9999999
        sum = 0.
        for f in self.terms: sum += f(r,z,t,v,l)
        return sum
    
# Class that represents a fluid
class fluid:
    # rest_density(kg/m3)
    # viscosity(Pa*s)
    # sound_speed(m/s)
    # bulk_modulus(kg/s2m)
    # specific_heat_p(J/kgK)
    # thermal_expansion(/K)
    # mean_excitation_energy(J)
    # molar_mass(u)
    # atomic_number()
    
    # Constructor
    def __init__(self,filename:str=None,rest_density:float=2966.3,viscosity:float=1.7e-2,sound_speed:float=653.47,bulk_modulus:float=1.2667e9, specific_heat:float=338.48,thermal_expansion:float=0.0013952,mean_excitation_energy:float=2.243e-18,molar_mass:float=131.293,atomic_number:float=54,name:str = 'Null'):
        
        self.name = name
        if filename is None:
            self.rest_density         = rest_density
            self.viscosity            = viscosity
            self.sound_speed          = sound_speed
            self.bulk_modulus         = bulk_modulus
            self.specific_heat        = specific_heat
            self.thermal_expansion    = thermal_expansion
            self.mean_excitation_energy = mean_excitation_energy
            self.molar_mass           = molar_mass
            self.atomic_number        = atomic_number
        else:
                        self.rest_density,self.viscosity,self.sound_speed,self.bulk_modulus,self.specific_heat,self.thermal_expansion,self.mean_excitation_energy,self.molar_mass,self.atomic_number = self.create_from_filename(filename)
    
    # Create Fluid from file
    def create_from_filename(self,filename:str='./data/fluids/LXE.txt'):
        file = open(filename)
        data = file.read()
        file.close()
        return [float(l.split(':')[-1]) for l in data.split('\n')[:9]]
    
    
    # Get Most probable energy deposition per unit length
    def max_energy(self,particle,x:float=1):
        v     = particle.speed
        beta  = v/c.c
        gamma = 1/np.sqrt(1-beta**2)
        K     = 0.307075 # MeV/mol cm^2
        A     = self.molar_mass # g/mol
#         mass_thickness = x * self.rest_density * 1e-3
        mass_thickness = x * 2.953
        ksi   = K * self.atomic_number * mass_thickness * particle.charge**2 / (2 * A * beta**2 * c.elementary_charge**2)
        
        return ksi * (np.log(2 * 0.510998928 * beta**2 * gamma**2 * ksi *1e12 / (self.mean_excitation_energy/c.eV)**2) - beta**2) / mass_thickness
#         return ksi * (np.log(2 * 0.510998928 * beta**2 * gamma**2 * ksi * 1e12 / (11.5)**2) - beta**2) / mass_thickness
        
    
    # Get the constant multiplier for the source term
    def source_multiplier(self):
        return - self.thermal_expansion/self.specific_heat
    
    # And the actual coefficient in front of the distribution of the thing
    def energy_deposition(self,particle,x:float=1):
        return self.source_multiplier() * particle.speed * self.max_energy(particle,x=x) * c.electron_volt * 1e6/x
    
    # Get the viscosity coefficient
    def viscosity_coefficient(self):
        return self.viscosity / self.bulk_modulus
        
# Particle class
class particle:
    # Constructor
    def __init__(self,filename:str = None, mass:float=1.883531627e-28,charge:float=-1.60217662e-19,speed:float=0.99*c.c,name:str = 'Null'):

        self.name = name
        
        if filename is None:
            self.mass   = mass
            self.charge = charge
            self.speed  = speed
            
        else:
            self.mass,self.charge,self.speed = self.create_from_filename(filename)
        
    # Create from filename
    def create_from_filename(self,filename:str='./data/particles/muon.txt'):
        file = open(filename)
        data = file.read()
        file.close()
        return [float(l.split(':')[-1]) for l in data.split('\n')[:3]]
    
    def __call__(self,v:float):
        self.speed = v
        
    
    
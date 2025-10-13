import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.polynomial import Polynomial

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18

# red surface
emax = Polynomial([1, -12, 6, 12, 9]).roots()[3].real
e = np.linspace(1/3, emax, 200)
a = (3*e**2 + 12*e - np.sqrt(9*e**4 + 72*e**3 - 42*e**2 + 24*e + 1) + 1)/(24*e)
e1_red = (1 - 3*a)*e/(a - e)
e2_red = (4*a**3*e - 4*a**2*e**2 - 6*a*e - a + 3*e)/(a - e)
e1_red[0], e2_red[0] = 1/3, -5/27

# cyan surface
e1 = np.linspace(1/3, 1/2, 200)
e2 = -2/9*np.sqrt(2)*e1*np.sqrt(3*e1 - 1) + (4/3)*e1 + (2/27)*np.sqrt(2)*np.sqrt(3*e1 - 1) - 17/27


# local models NSI paper
E1LM = np.arange(0.175, 0.5125, 0.0125)
E2LM = [-3.33333333e-01, -3.19659055e-01 ,-3.05925365e-01, -2.92205698e-01, 
        -2.78747859e-01 ,-2.65342926e-01 ,-2.52950581e-01 ,-2.40673388e-01, 
        -2.29472786e-01 ,-2.18933024e-01 ,-2.08318774e-01 ,-1.96822165e-01, 
        -1.87247622e-01 ,-1.76528844e-01 ,-1.62336664e-01 ,-1.48248986e-01, 
        -1.34014435e-01 ,-1.19780379e-01 ,-1.05666008e-01 ,-9.17544873e-02, 
        -7.80700339e-02 ,-6.45965142e-02 ,-5.13208155e-02 ,-3.82320057e-02, 
        -2.53207881e-02 ,-1.25791301e-02, -6.03319070e-08]

data = np.loadtxt('e1e2_nsi_boundary_old.txt')
E1NSI = data[:,0]
E2NSI = data[:,1]

# NSI with higher inflations
data = np.loadtxt('e1e2_nsi_boundary_improved.txt')

# Clean up the points where the optimization failed
take = [True] + ((data[:-1,1] == data[1:,1])
                 + (data[:-1,1] <= data[1:,1] - 1e-3)).tolist()
NSIbetter = data[take,:].T
for _ in range(5):
        take = [True] + ((NSIbetter[1,:-1] == NSIbetter[1,1:])
                         + (NSIbetter[1,:-1] <= NSIbetter[1,1:]-1e-3)).tolist()
        NSIbetter = NSIbetter[:,take]
        
E1NSIbetter = NSIbetter[:,0]
E2NSIbetter = NSIbetter[:,1]

# positivity constraints
E1_positivity_constraints = np.linspace(0.15, 0.5, 100)
E2_positivity_constraints = np.maximum(-1/3, 2*E1_positivity_constraints - 1)

plt.fill_between([0, 0.1], [0, 0.1], [0, 0.1], color='white', label=r'$\bigtriangleup$-local [27]')

plt.fill_between(E1LM, [-0.34]*len(E1LM), E2LM, color='#9048E1', label=r'$\bigtriangleup$-local') #purple

plt.fill_between(e1_red, [-0.34]*len(e1_red), e2_red, color='#FFDD69', label='Unknown') #yellow
plt.fill_between(e1, [-0.34]*len(e1), e2, color='#FFDD69') #yellow

plt.fill_between(E1NSIbetter, [-0.34]*len(E1NSIbetter), E2NSIbetter,
                color='#6ACCD0', label=r'Non-$\bigtriangleup$') #cyan

plt.plot(E1NSI, E2NSI, 'k-', linewidth=1, label=r'Non-$\bigtriangleup$ [27]')

plt.fill_between(E1_positivity_constraints,
                 [-0.35]* len(E1_positivity_constraints),
                 E2_positivity_constraints, color='#BFBFBF', label='Non-dist.') #gray
plt.plot(1/3, -5/27, 'ko', markersize=3, fillstyle='none') # interesting point

plt.ylim(-0.35, 0.02)
plt.xlim(0.15, 0.5)
plt.xlabel(r'$E_1$')
plt.ylabel(r'$E_2$')

plt.legend()

plt.savefig('NSI_comparison.pdf', bbox_inches='tight')
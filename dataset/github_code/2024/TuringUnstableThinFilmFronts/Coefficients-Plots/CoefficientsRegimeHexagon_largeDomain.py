from sympy import *
from sympy.parsing.mathematica import parse_mathematica
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from Difference_M import Mdiff

b, g = var('b g')

# specify colors
tumblue = '#64A0C8'
tumorange = '#E37222'
tumgreen = '#A2AD00'
tumivory = '#DAD7CB'

######### read mathematica expression from external files and convert to python expressions ##########
print("loading mathematica expressions\n")
# load self coefficient #
with open('Self-cubic-coeffs-hex.txt','r') as file:
    K0 = file.read()

K0 = K0.replace('{','')
K0 = K0.replace('}','')
K0_python = parse_mathematica(K0)

K0_fast_eval = lambdify([b,g],K0_python)

# load cross coefficient square
with open('cross-cubic-coeffs-hex.txt','r') as file:
    K2 = file.read()

K2 = K2.replace('{','')
K2 = K2.replace('}','')
K2_python = parse_mathematica(K2)

K2_fast_eval = lambdify([b,g],K2_python)    

############## generate grid ##############
nbeta = 2000
ng = 1000
betaGrid = np.linspace(0.01,50,nbeta)
gGrid = np.linspace(0.1,20,ng)
X , Y = np.meshgrid(betaGrid,gGrid,indexing='xy')
existence = np.zeros((len(gGrid),len(betaGrid)))
Mdifference = np.zeros((len(gGrid),len(betaGrid)))
k0 = np.zeros((len(gGrid),len(betaGrid)))
k2 = np.zeros((len(gGrid),len(betaGrid)))

# define function for Mm*-Mo* to determine if points are in the relevant parameter area
MmMinusMo = Mdiff()

############## evaluate function on grid ##############
# write z-values by evaluating kappajj
for ii in range(len(betaGrid)):
    print(str(ii)+"/"+str(len(betaGrid)))
    for jj in range(len(gGrid)):
        k2[jj,ii] = K2_fast_eval(X[jj,ii],Y[jj,ii])
        Mdifference[jj,ii] = MmMinusMo(X[jj,ii],Y[jj,ii])
        if Mdifference[jj,ii] < 0:
            k2[jj,ii] = np.nan

# plot and save results
# coefficients: K2
fig, ax = plt.subplots()
densityK2 = ax.pcolormesh(X,Y,k2,cmap='jet')
ax.contour(X,Y,k2,[0],colors='w',linestyles='dashed')
fig.colorbar(densityK2)

plt.title(r'$K_2$')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("DensityMapK2LargeDomain.png",dpi=500)
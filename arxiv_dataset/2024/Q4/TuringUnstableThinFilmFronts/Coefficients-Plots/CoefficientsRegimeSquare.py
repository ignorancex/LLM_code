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
with open('cross-cubic-coeffs-square.txt','r') as file:
    K1 = file.read()

K1 = K1.replace('{','')
K1 = K1.replace('}','')
K1_python = parse_mathematica(K1)

K1_fast_eval = lambdify([b,g],K1_python)    

############## generate grid ##############
nbeta = 2500
ng = 2500
betaGrid = np.linspace(0.01,10,nbeta)
gGrid = np.linspace(0.1,20,ng)
X , Y = np.meshgrid(betaGrid,gGrid,indexing='xy')
existence = np.zeros((len(gGrid),len(betaGrid)))
selection = np.zeros((len(gGrid),len(betaGrid)))
selectionSubcritical = np.zeros((len(gGrid),len(betaGrid)))
k0 = np.zeros((len(gGrid),len(betaGrid)))
k1 = np.zeros((len(gGrid),len(betaGrid)))

# define function for Mm*-Mo* to determine if points are in the relevant parameter area
MmMinusMo = Mdiff()

############## evaluate function on grid ##############
# write z-values by evaluating kappajj
for ii in range(len(betaGrid)):
    print(str(ii)+"/"+str(len(betaGrid)))
    for jj in range(len(gGrid)):
        k0[jj,ii] = K0_fast_eval(X[jj,ii],Y[jj,ii])
        k1[jj,ii] = K1_fast_eval(X[jj,ii],Y[jj,ii])
        if MmMinusMo(X[jj,ii],Y[jj,ii]) >= 0:
            existence[jj,ii] = np.heaviside(-k0[jj,ii],0) + 2*np.heaviside(-k0[jj,ii]-k1[jj,ii],0) + 1
            # selection = 1 if K1-K0 > 0, = -1 if K1-K0 < 0, and 0 otherwise if not both patterns exist for M0 > 0
            selection[jj,ii] = np.sign((k1[jj,ii]-k0[jj,ii]))*2*np.heaviside((existence[jj,ii]-3.5),0)
            # selectionSubcritical = 1 if K1-K0 > 0, = -1 if K1-K0 < 0, and 0 otherwise if noth both patterns exist for M0 < 0
            selectionSubcritical[jj,ii] = np.sign((k1[jj,ii]-k0[jj,ii]))*np.heaviside(k1[jj,ii]+k0[jj,ii],0)*np.heaviside(k0[jj,ii],0)
        else:
            existence[jj,ii] = 0
            selection[jj,ii] = -2

# plot and save results
# existence
fig, ax = plt.subplots()
cs_existence = ax.pcolormesh(X,Y,existence,cmap=mpl.colors.ListedColormap([tumblue,tumgreen,tumorange]))
proxy = [plt.Rectangle((0, 0), 1, 1, fc=fc) for fc in [tumblue,tumgreen,tumorange]]
plt.legend(proxy, ["none", "rolls", "squares & rolls"])

plt.title('Existence for square lattice')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("ExistenceSquareLattice.png",dpi=500)

# selection supercritical (M0 > 0)
fig, ax = plt.subplots()
cs_selection = ax.pcolormesh(X,Y,selection,cmap=mpl.colors.ListedColormap([tumgreen,tumblue,tumorange]))
proxy = [plt.Rectangle((0, 0), 1, 1, fc=fc) for fc in [tumgreen,tumblue,tumorange]]
plt.legend(proxy, ["rolls","no selection", "squares"])
# plt.show()

plt.title(r'Selection on square lattice ($M_0 > 0$)')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("SelectionSquareLatticeSupercrit.png",dpi=500)

# selection subcritical (M0 < 0)
fig, ax = plt.subplots()
cs_selection = ax.pcolormesh(X,Y,selectionSubcritical,cmap=mpl.colors.ListedColormap([tumgreen,tumblue,tumorange]))
proxy = [plt.Rectangle((0, 0), 1, 1, fc=fc) for fc in [tumgreen,tumblue,tumorange]]
plt.legend(proxy, ["rolls","no selection", "squares"])
# plt.show()

plt.title(r'Selection on square lattice ($M_0 < 0$)')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("SelectionSquareLatticeSubcrit.png",dpi=500)

# coefficients: K0
fig, ax = plt.subplots()
densityK0 = ax.pcolormesh(X,Y,k0,cmap='jet')
# ax.contour(X,Y,k0,[0],colors='w',linestyles='dashed')
ax.contour(X,Y,k0,[0],colors='w')
fig.colorbar(densityK0)

plt.title(r'$K_0$')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("DensityMapK0.png",dpi=500)

# coefficients: K1
fig, ax = plt.subplots()
densityK1 = ax.pcolormesh(X,Y,k1,cmap='jet')
# ax.contour(X,Y,k1,[0],colors='w',linestyles='dashed')
ax.contour(X,Y,k1,[0],colors='w')
fig.colorbar(densityK1)

plt.title(r'$K_1$')
plt.xlabel('β')
plt.ylabel('g')
plt.savefig("DensityMapK1.png",dpi=500)

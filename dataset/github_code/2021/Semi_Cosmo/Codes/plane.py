#Plotting the fundamental plane
#Note: this file already has the all the coordinates of the points and just plots them
#Author: Shashank Dattathri

import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import scipy

plt.rc('lines', linewidth=2, color='r')
plt.rcParams.update({'font.size': 16})

def f(x,a,b,c,d):
    return a+b*x[0]+c*x[1]+d*x[2]

def f_opt(x1,x2,x3):
    a=7.15144748
    b=0.25724271
    c=0.57645242
    d=0.75098742
    return a+b*x1+c*x2+d*x3

unitrho=1.67262171e-27
unitl=3.0856775807e21
unitv=1.e7
unitmdot=unitrho*unitl**2*unitv
msun=2.e33
yr=3.145e7
unitmdot=unitmdot/msun*yr

Mhalo=[]
Mbh=[1.643779e+40, 1.482706e+40, 3.708983e+39, 2.380403e+39, 
    8.488408e+40, 1.517865e+40, 7.21488e+39, 3.157966e+39, 
    2.157925e+42, 9.025898e+41, 3.749555e+41, 6.763282e+40,
    1.18194e+43, 3.822347e+42, 1.647974e+42, 2.2732724e+41,
    1.010051e+42,4.181187e+41, 3.870082e+41, 7.15764e+40, 4.204154e+40,1.159769e+40]
epsilonarray=[]
colors=[]

# k for cooling, r,y,g,b,c for 1e-3,1e-4,1e-5,1e-6,1e-7

epsilonarray.append(1e-4)
epsilonarray.append(1e-5)
epsilonarray.append(1e-6)
epsilonarray.append(1e-7)
colors.append('y')
colors.append('g')
colors.append('b')
colors.append('c')
for i in range(4):
    Mhalo.append(1e12)

epsilonarray.append(1e-4)
epsilonarray.append(1e-5)
epsilonarray.append(1e-6)
epsilonarray.append(1e-7)
colors.append('y')
colors.append('g')
colors.append('b')
colors.append('c')
for i in range(4):
    Mhalo.append(1e13)

epsilonarray.append(1e-3)
epsilonarray.append(1e-4)
epsilonarray.append(1e-5)
epsilonarray.append(1e-6)
colors.append('r')
colors.append('y')
colors.append('g')
colors.append('b')
for i in range(4):
    Mhalo.append(1e14)

epsilonarray.append(1e-3)
epsilonarray.append(1e-4)
epsilonarray.append(1e-5)
epsilonarray.append(1e-6)
colors.append('r')
colors.append('y')
colors.append('g')
colors.append('b')
for i in range(4):
    Mhalo.append(1e15)


epsilonarray.append(1e-4)
epsilonarray.append(1e-5)
epsilonarray.append(1e-5)
epsilonarray.append(1e-6)
epsilonarray.append(1e-6)
epsilonarray.append(1e-7)
colors.append('y')
colors.append('g')
colors.append('g')
colors.append('b')
colors.append('b')
colors.append('c')
for i in range(6):
    Mhalo.append(1e14)

Mdotin=[-0.7428903230581495,-3.116564144932948,-6.561460355624479,-15.83859024774076,
-2.066132309688745,-6.955169841596643,-20.168159312742763,-22.077602537705335,
-4.800606501226103,-42.062021231288135,-142.860866607784,-254.24170024714235,
-20.9329767332638,-193.969564002448,-515.722455821421,-598.201958738391,
-42.55375590180314,-159.32377848802992,-140.9603758265391,-198.94537469919734,-170.9586531638193,-252.4578621]

Mdotin=Mdotin*(-1*np.ones(len(Mdotin)))
    
fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(np.log10(Mhalo),np.log10(Mdotin),np.log10(Mbh/(msun*np.ones(len(Mbh)))),color=colors)
ax.set_zlabel(r"$\log(M_{\rm BH,z=0})$",labelpad=18)
ax.set_xlabel(r"$\log(M_0)$",labelpad=18)
ax.set_ylabel(r"$\log(\dot{M}_{\rm in})$",labelpad=18)

xdata=[]
ydata=[]

r_patch = mpatches.Patch(color='r', label="$\epsilon=10^{-3}$")
y_patch = mpatches.Patch(color='y', label="$\epsilon=10^{-4}$")
g_patch = mpatches.Patch(color='g', label="$\epsilon=10^{-5}$")
b_patch = mpatches.Patch(color='b', label="$\epsilon=10^{-6}$")
c_patch = mpatches.Patch(color='c', label="$\epsilon=10^{-7}$")
plt.legend(handles=[r_patch,y_patch,g_patch,b_patch,c_patch])

halo_array=np.linspace(12,15,10)
mdotin_array=np.linspace(-1,3,10)

xx,yy=np.meshgrid(halo_array,mdotin_array)
mbh_array_1em7=f_opt(xx,yy,-7)
mbh_array_1em6=f_opt(xx,yy,-6)
mbh_array_1em5=f_opt(xx,yy,-5)
mbh_array_1em4=f_opt(xx,yy,-4)
mbh_array_1em3=f_opt(xx,yy,-3)
ax.plot_surface(xx, yy, mbh_array_1em7,alpha=0.4,color='c')
ax.plot_surface(xx, yy, mbh_array_1em6,alpha=0.4,color='b')
ax.plot_surface(xx, yy, mbh_array_1em5,alpha=0.4,color='g')
ax.plot_surface(xx, yy, mbh_array_1em4,alpha=0.4,color='y')
ax.plot_surface(xx, yy, mbh_array_1em3,alpha=0.4,color='r')

ax.set_zlim(5.8,10.2)
plt.show()

xdata=[np.log10(Mhalo),np.log10(Mdotin),np.log10(epsilonarray)]
ydata=np.log10(Mbh/(msun*np.ones(len(Mbh))))


popt, pcov = curve_fit(f, xdata, ydata)
print("Best fit parameters:")
print(popt)

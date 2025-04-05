import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
import sys

mu = float(sys.argv[1])
ebin = float(sys.argv[2])
abin = float(sys.argv[3])
ip = float(sys.argv[4])

data = np.genfromtxt("a_crit_Incl[%i].txt" % ip,delimiter=',',comments='#')

X = data[:,0]
Y = data[:,1]
Z = data[:,2]

xi = np.concatenate(([0.001],np.arange(0.01,1,0.01),[0.999]))
yi = np.arange(0,0.81,0.01)
zi = griddata((X,Y),Z,(xi[None,:],yi[:,None]),method = 'linear',fill_value=0)

f = interp2d(xi, yi, zi, kind='linear')

astab = f(mu,ebin)[0]
print "a_c = %1.2f AU" % (astab*abin)
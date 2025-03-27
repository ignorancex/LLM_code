#Mass accretion histories of the dark matter halos as given by van den Bosch et al (2014)
#Author: Shashank Dattathri

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.ticker import FormatStrFormatter

plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 14})

x2ticksarray=[]


data1=np.loadtxt('1e11_mah.dat')
data1=np.transpose(data1)
data2=np.loadtxt('1e12_mah.dat')
data2=np.transpose(data2)
data3=np.loadtxt('1e13_mah.dat')
data3=np.transpose(data3)
data4=np.loadtxt('1e14_mah.dat')
data4=np.transpose(data4)
data5=np.loadtxt('1e15_mah.dat')
data5=np.transpose(data5)

x1=np.log10(1+data1[1])
x2=data1[2]


fig, ax = plt.subplots()
ax.plot(x1,data1[3],label="$M_0=10^{11} M_\odot$")
ax.plot(x1,data2[3],label="$M_0=10^{12} M_\odot$")
ax.plot(x1,data3[3],label="$M_0=10^{13} M_\odot$")
ax.plot(x1,data4[3],label="$M_0=10^{14} M_\odot$")
ax.plot(x1,data5[3],label="$M_0=10^{15} M_\odot$")
ax.plot(np.log10(1.+6.)*np.ones(100),np.linspace(-5,0,100),'k--')
ax.plot(np.linspace(0,1,100),np.ones(100)*np.log10(0.04),'k--')
ax.text(0.75,-4.5," $z=6$\n$(t=0)$")
ax.text(0.1,-1.75,"$M(z)/M_0=0.04$")
ax.set_xbound(0,1)
ax.set_ybound(-5,0)

ax.set_xlabel("$\log_{10}(1+z)$")
ax.set_ylabel("$\log_{10}(M(z)/M_0)$")

x1labelpoints=10**(ax.get_xticks())-1
x2labelpoints=[0., 5.675, 9.479, 11.624, 12.734, 13.296]

ax2=ax.twiny()
ax2.set_xticks( ax.get_xticks() )
ax2.set_xbound(ax.get_xbound())

ax2.set_xticklabels([x2labelpoints[i] for i in range(len(ax.get_xticks()))])

ax2.set_xlabel("Lookback time (Gyr)")

ax.legend()
plt.show()
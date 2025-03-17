import numpy as np
import matplotlib.pyplot as plt
import pandas
import h5py
import greedy_contours
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV
from MTOV_R14_priors import p_MTOV_R14
from fjet.fitting_formulae import Mdisk_KF20 as MD_K
from fjet.fitting_formulae import Mej_KF20 as Mej_K
from fjet.utils import M_HMNS,M_GRB

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

def itp_PGRB(m1,m2,m10,m20,PGRB):
    j1 = np.searchsorted(m20,m2)
    j0 = j1-1
    p1 = np.interp(m1,m10,PGRB[j1])
    p0 = np.interp(m1,m10,PGRB[j0])
    return p0 + (m2-m20[j0])/(m20[j1]-m20[j0])*(p1-p0)


# grid
R14 = np.linspace(9.,15.,50)
MTOV = np.linspace(1.8,2.5,51)
XID = np.linspace(0.01,5.,49)

pi_r_mtov_xid = np.zeros([len(R14),len(MTOV),len(XID)])

for i in range(len(R14)):
    for j in range(len(XID)):
        pi_r_mtov_xid[i,:,j] = p_MTOV_R14(MTOV,R14[i],xi_d=XID[j])*np.exp(-0.5*(XID[j]-1.)/0.5)

pi_r_mtov_xid/=np.trapz(np.trapz(np.trapz(pi_r_mtov_xid,XID,axis=2),MTOV,axis=1),R14,axis=0)

m10 = np.linspace(1.,2.3,100)
m20 = np.linspace(1.,2.3,101)

PGRB = np.zeros([len(m20),len(m10)])

for i in range(len(m10)):
    for j in range(len(m20)):
        M = m10[i]+m20[j]
        nu = m10[i]*m20[j]/(m10[i]+m20[j])**2
        THj = np.zeros(pi_r_mtov_xid.shape)
        for k in range(len(R14)):
            
            C2 = 1.48*m20[j]/R14[k]
            C1 = 1.48*m10[i]/R14[k]
            mgw = 0.25*nu*1.48*M**2/R14[k]
            md = MD_K(m20[j],C2)*XID.reshape([1,len(XID)])
            mej = Mej_K(m10[i],m20[j],C1,C2)
            mrem = M-mgw-md #-mej
            THj[k] = (mrem>=(1.2*MTOV.reshape([len(MTOV),1])))*(md>=1e-3)
            
        PGRB[j,i] = np.trapz(np.trapz(np.trapz(pi_r_mtov_xid*THj,XID,axis=2),MTOV,axis=1),R14,axis=0)
    
cmap=plt.cm.viridis
cs = plt.contourf(m10,m20,PGRB,levels=np.linspace(0,1,21),cmap=cmap,zorder=0)
for c in cs.collections:
    c.set_rasterized(True)

plt.colorbar(label=r'$P_\mathrm{j}(M_1,M_2)$',ticks=np.linspace(0.,1.,11))


plt.fill([1.,2.2,1.],[1.,2.2,2.2],edgecolor='k',facecolor='w',zorder=1)
plt.fill([1.,2.2,2.2],[1.,1.,2.2],edgecolor='k',facecolor=cmap(0),zorder=-1)


Gdata = np.load('data/GalacticDNSMassSamples.npy')
merging = np.arange(10)
ma = Gdata[merging,:,0]
mb = Gdata[merging,:,1]

for i in range(len(merging)):
    m1 = np.where(ma[i]>=mb[i],ma[i],mb[i])
    m2 = np.where(ma[i]>=mb[i],mb[i],ma[i])
    c2,b1,b2 = greedy_contours.samples_to_mesh(m1,m2,bins=[100,101])
    plt.contour(b1,b2,c2,levels=[0.9],colors='w',linewidths=0.5,zorder=100)
    plt.plot([np.mean(m1)],[np.mean(m2)],marker='o',ls='None',c='w',markersize=3,markeredgecolor='k',zorder=100)

plt.plot([0],[0],marker='o',ls='None',c='w',markersize=3,markeredgecolor='k',label=r'Galactic BNS ($t_\mathrm{merge}<t_\mathrm{H}$)')
#plt.annotate(xy=(1.74,1.13),ha='left',va='bottom',text='J1913+1102',color='w')

m1_G17,m2_G17 = np.load('data/GW170817_m1m2_posterior_samples_LS.npy')
G17_c,m1_G17_c,m2_G17_c = greedy_contours.samples_to_mesh(m1_G17,m2_G17,bins=(65,66))


# compute Pjet(GW170817)
P17 = np.zeros(len(m1_G17))

for i in range(len(P17)):
    P17[i] = itp_PGRB(m1_G17[i],m2_G17[i],m10,m20,PGRB)


print('Pjet(GW170817) = {0:.3g}'.format(np.mean(P17)))


mask = (m1_G17_c<m2_G17_c)
plt.contour(m1_G17_c,m2_G17_c,np.ma.masked_array(G17_c,mask=mask),levels=[0.5,0.9],colors='#FFBC00',zorder=3,linewidths=0.8)
plt.plot([0],[0],ls='-',c='orange',label='GW170817')


m1_G19,m2_G19 = np.load('data/GW190425_m1m2_posterior_samples_LS.npy')
G19_c,m1_G19_c,m2_G19_c = greedy_contours.samples_to_mesh(m1_G19,m2_G19,bins=(65,66))

mask = (m1_G19_c<m2_G19_c)
plt.contour(m1_G19_c,m2_G19_c,np.ma.masked_array(G19_c,mask=mask),levels=[0.5,0.9],colors='#18FF00',zorder=3,linewidths=0.8)
plt.plot([0],[0],ls='-',c='#00CC00',label='GW190425')

plt.xlim([1.1,2.])
plt.ylim([1.1,2.])

plt.xlabel(r'$M_1\,[\mathrm{M_\odot}]$')
plt.ylabel(r'$M_2\,[\mathrm{M_\odot}]$')


plt.tick_params(which='both',direction='in',top=True,right=True)

plt.legend(loc='upper left',frameon=False)

# plt.plot([1.26],[1.26],marker='*',ls='None',markeredgecolor='purple',color='w')
# plt.annotate(xy=(1.26,1.26),text='ECSN',va='top',ha='left',color='w')


plt.savefig('../figures/Pjet_m1m2.pdf')

plt.show()


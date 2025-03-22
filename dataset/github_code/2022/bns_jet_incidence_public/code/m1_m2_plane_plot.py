import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect
import greedy_contours
import pandas
import h5py
import fjet
from fjet import NS
from fjet.fitting_formulae import Mb

recomp_gaussian = False
recomp_uniform = False
plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=8.,3.
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

def Mrem(m1,m2,md,mej,EoS):
    R14 = NS.R(1.4,EoS)
    m1b = Mb(m1,R14=R14,P_PK=100.)
    m2b = Mb(m2,R14=R14,P_PK=100.)
    mbrem = m1b+m2b-md-mej
    def fz(mg):
        return Mb(mg,R14=NS.R(NS.M_max[EoS],EoS),P_PK=0.5)-mbrem
    
    mrem = bisect(fz,0.5*mbrem,mbrem*1.5)
    
    return mrem
    
    


Mdisk_min = 1e-3

# ---------------------- M1,M2 plane -----------------------


m1_G17,m2_G17 = np.load('data/GW170817_m1m2_posterior_samples_LS.npy')
G17_c,m1_G17_c,m2_G17_c = greedy_contours.samples_to_mesh(m1_G17,m2_G17,bins=(80,81))

m1_G19,m2_G19 = np.load('data/GW190425_m1m2_posterior_samples_LS.npy')
G19_c,m1_G19_c,m2_G19_c = greedy_contours.samples_to_mesh(m1_G19,m2_G19,bins=(80,81))


hpad = 0.075
vpad = 0.025
cw = 0.02*0
pw = 1./3. - cw/3. - 1.*hpad/3.
ph = 0.85

plt.figure(figsize=(8.,3.))

for i_EoS,EoS in enumerate(['APR4','SFHo','DD2']):
    
    m1 = np.linspace(1.,2.5,200)
    m2 = np.linspace(1.,2.5,200)
    l1 = NS.Lambda(m1,EoS)
    l2 = NS.Lambda(m2,EoS)
    C1 = NS.C(m1,EoS)
    C2 = NS.C(m2,EoS)
    mrem = np.zeros([len(m1),len(m2)])
    tidal = np.zeros([len(m1),len(m2)])
    mt = np.zeros([len(m1),len(m2)])
    R16 = NS.R(1.6,EoS)
    Rmax = NS.R(NS.M_max[EoS],EoS)
    md = np.zeros([len(m1),len(m2)])
    mej = np.zeros([len(m1),len(m2)])
    
    for i in range(len(m1)):
    
        lt = fjet.utils.lambda_tilde(m1[i],l1[i],m2,l2)
        
        mgw = np.zeros_like(m2)
        mgw[m1[i]>=m2] = fjet.bns_lum.egw(m1[i],m2[m1[i]>=m2],l1[i],l2[m1[i]>=m2])
        mgw[m1[i]<m2] = fjet.bns_lum.egw(m2[m1[i]<m2],m1[i],l2[m1[i]<m2],l1[i])
        md[i] = fjet.fitting_formulae.Mdisk_KF20(np.where(m1[i]>m2,m2,m1[i]),np.where(m1[i]>m2,C2,C1[i]))
        #md[i] = fjet.fitting_formulae.Mdisk_B21(lt,m1[i],m2)
        mej[i] = fjet.fitting_formulae.Mej_KF20(np.where(m1[i]>m2,m1[i],m2),np.where(m1[i]>m2,m2,m1[i]),np.where(m1[i]>m2,C1[i],C2),np.where(m1[i]>m2,C2,C1[i]))
        mrem[i] = m1[i]+m2-mgw-md[i]-mej[i]
        #mrem[i]=np.array([Mrem(m1[i],m2[j],md[i,j],mej[i,j],EoS) for j in range(len(m2))])
        q = np.minimum(m2/m1[i],m1[i]/m2)
        mt[i] = fjet.fitting_formulae.Mthres_Bauswein21(q,NS.M_max[EoS],R16)-m1[i]-m2
    
    
    s = 0.75
    mrem = gaussian_filter(mrem,s)
    md = gaussian_filter(md,s)
    mej = gaussian_filter(mej,s)
    
    
    m1_m,m2_m = np.meshgrid(m1,m2)
    q_m = m2_m/m1_m
    q_1 = m1_m<m2_m
    stable_NS = np.ma.masked_array(mrem<NS.M_max[EoS],mask=~q_1)
    SMNS = np.ma.masked_array((mrem<(1.2*NS.M_max[EoS]))&(~stable_NS),mask=~q_1)
    HMNS = np.ma.masked_array((mrem>=1.2*NS.M_max[EoS])&(mt>0.),mask=~q_1)
    BH = np.ma.masked_array((mt<=0.)&(mrem>=1.2*NS.M_max[EoS]),mask=~q_1)
    GRB = np.ma.masked_array((mrem>=1.2*NS.M_max[EoS])&(md>Mdisk_min),mask=~q_1)
    BHNS = np.ma.masked_array((m1_m>NS.M_max[EoS])|(m2_m>NS.M_max[EoS]),mask=~q_1)
    
    
    ax = plt.axes([hpad + pw*i_EoS,1.-ph,pw-0.018,ph - 3*vpad])
    plt.title(EoS)
    
    # regions
    reg_names = ['NS','SMNS','HMNS','BH','Jet','']
    reg = [stable_NS,SMNS,HMNS,BH,GRB,BHNS]
    colors = ['#f1eef6','#d7b5d8','#df65b0','#ce1256','k','#CE1256']
    textcolors = ['grey','#d7b5d8','#df65b0','#ce1256','k','grey']
    fill = [True,True,True,True,False,True]
    hatch = [None,None,None,None,None,'\\\\']
    
    for i in range(len(reg_names)):
        if fill[i]:
            cs = plt.contourf(m1_m,m2_m,reg[i].T,levels=(0.5,1.),colors=[colors[i]],zorder=1,hatches=[hatch[i]])
            if hatch[i] is not None:
                cs.collections[0].set_edgecolor('grey')
                cs.collections[0].set_linewidth(0.)
                
        else:
            plt.contour(m1_m,m2_m,reg[i].T,levels=[0.5],colors=['k'],linestyles=['--'],zorder=4)
        
        bary_m1 = np.mean(m1_m[reg[i].T&(~q_1)])
        bary_m2 = np.mean(m2_m[reg[i].T&(~q_1)])
        
        bary_eq = 0.5*(bary_m1+bary_m2)
        
        if reg_names[i]!='Jet':
            xy = (bary_eq,bary_eq)
        else:
            xy = (bary_m1,bary_m2)
    
        plt.annotate(xy=xy,text=reg_names[i],color=textcolors[i],va='bottom',ha='right')
    

    # plot GW170817 contours
    mask = (m1_G17_c<m2_G17_c)
    plt.contour(m1_G17_c,m2_G17_c,np.ma.masked_array(G17_c,mask=mask),levels=[0.5,0.9],colors='#FFBC00',zorder=3,linewidths=0.5)
    
    if i_EoS==1:
        plt.annotate(xy=(1.7,1.1),text='GW170817',ha='left',color='#FFBC00')
    
    # plot GW190425 contours
    mask = (m1_G19_c<m2_G19_c)
    plt.contour(m1_G19_c,m2_G19_c,np.ma.masked_array(G19_c,mask=mask),levels=[0.5,0.9],colors='#18FF00',zorder=3,linewidths=0.5)
    
    if i_EoS==1:
        plt.annotate(xy=(1.81,1.6),text='GW190425',ha='left',color='#18FF00')
    
    # plot galactic pulsars
    Gdata = np.load('data/GalacticDNSMassSamples.npy')
    merging = np.arange(10)
    ma = Gdata[merging,:,0]
    mb = Gdata[merging,:,1]
    
    for i in range(len(merging)):
        m1 = np.where(ma[i]>=mb[i],ma[i],mb[i])
        m2 = np.where(ma[i]>=mb[i],mb[i],ma[i])
        c2,b1,b2 = greedy_contours.samples_to_mesh(m1,m2,bins=[100,101])
        plt.contour(b1,b2,c2,levels=[0.9],colors='w',linewidths=0.5)
        plt.plot([np.mean(m1)],[np.mean(m2)],marker='o',ls='None',c='w',markersize=3,markeredgecolor='k',markeredgewidth=0.5,zorder=4)
    
    plt.plot([0],[0],marker='o',ls='None',c='w',markersize=3,markeredgecolor='k',markeredgewidth=0.5,label=r'Galactic BNS ($t_\mathrm{merge}<t_\mathrm{H}$)')
    # plt.annotate(xy=(1.68,1.25),ha='left',va='center',text='J1913+1102',color='r')
    
    plt.xlabel(r'$M_1$ [M$_\odot$]')
    
    if i_EoS==0:
        plt.ylabel(r'$M_2$ [M$_\odot$]')
    
    plt.xlim(1.,2.3)
    plt.ylim(1.,2.3)
    
    plt.tick_params(which='both',direction='in',top=True,right=True,labelleft=(i_EoS==0))
    
    # plot approx method GRB border
    q = np.linspace(0.5,1.,100)
    MH = np.zeros_like(q)
    Mnd = np.zeros_like(q)
    for i in range(len(q)):
        MH[i] = fjet.utils.M_GRB(q[i],(NS.M_max[EoS],NS.R(1.4,EoS)))
        Mnd[i] = fjet.utils.M_nodisk(q[i],(NS.M_max[EoS],NS.R(1.4,EoS)))
    
    M2H = MH/(1.+q**-1)
    M1H = MH/(1.+q)
    M2nd = Mnd/(1.+q**-1)
    M1nd = Mnd/(1.+q)
    
    plt.plot([0],[0],'--k',lw=1,label=r'$R=R(M)$')
    plt.plot(M1H,M2H,'--b',lw=1,zorder=3.5,label=r'$R=R_{1.4}$')
    plt.plot(M1nd,M2nd,'--b',lw=1,zorder=3.5)
    
    if i_EoS==1:
        plt.legend(frameon=False,loc='upper left')


plt.savefig('../figures/M1_M2_plane.pdf')
    
plt.show()

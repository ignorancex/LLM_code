#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 01:28:47 2023

@author: shin
"""

# %%
import numpy as np
from scipy import linalg ,optimize , interpolate ,integrate
from optparse import OptionParser
from collections import OrderedDict
import pickle
import astropy.units as u
from astropy.coordinates import angular_separation as ang 

import pandas as pd
import time
import ray 
a=time.time()
c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

usage = 'usage: %prog [options]'
parser = OptionParser(usage)
parser.add_option( "-d", "--details", action="store", type="int", default=2, dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option( "-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option( "-s", "--scan", action = "store_true", default=False, dest="SCAN", help = "Whether to do a scan")
parser.add_option( "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell. ")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole too?")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO?")
parser.add_option( "-e","--evaluate", action = "store",type='int', dest="EVAL", default=3, help = "What evaluation needs to be done, 1: LCDM , 2: z Taylor , 3: z Dipole Taylor , 4: H0 dipole, 5: H0,Q0 dipole ,6: Quadrupolar ?")

parser.add_option( "--zindex", action = "store", type='int',default=9, dest="ZINDEX", help = "ZINDEX?")
parser.add_option( "--parallel", action = "store_true", default=True, dest="PARALLEL", help = "PARALELLIZE USING RAY")

parser.add_option("--zlim", action = "store",type="float", default=0.00937, dest="ZLIM")
(options, args) = parser.parse_args()

dctdet={1:"Non scalar",2:"Exponential"}

if options.DET==1:
    STYPE='NoScDep'
elif options.DET==2:
    STYPE='Exp'

else:
    STYPE='None'







CMBdipdec = -7
CMBdipra = 168
    

zlim=0.00937
# %%
# %%
CMBdipdec = -7
CMBdipra = 168
if options.MET==1:
    met='Nelder-Mead'
elif options.MET==2:
    met='SLSQP'
elif options.MET==3:
    met='Powell'
elif options.MET==7:
    met="L-BFGS-B"
elif options.MET==4:
    met="trust-constr"
#print("Method Used:",met)



CMBdipdec = -7
CMBdipra = 168
two_mppra=164.6877348*u.deg 
two_mppdec=-17.1115932*u.deg
raLG_SUN= 333.53277784*u.deg
decLG_SUN = 49.33111122*u.deg

vLG_SUN= 299 #km/s


if options.PARALLEL:
    ray.init()
@ray.remote
def Minimizer(qm0=None,qd0=None,zlim=options.ZLIM):
    Z = pd.read_csv( '../Z_mbcorr.csv' )
    df=pd.read_csv('../Pantheon+SH0ES.dat',delimiter=' ')
    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
    zDF=np.sqrt((1-vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c)/(1+vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c))-1
    zLG=((1+df['zHEL'])/(1+zDF))-1
    Z['zLG']=zLG
    Z=Z[Z['zHEL']>zlim]
    Z=Z.sort_values('zHEL')
    name=" "
    n=np.linspace(0,1700,1701).astype('int')
    
    if options.REVB:
        print ('reversing bias')
        Z['m_b_corr'] = Z['m_b_corr'] + Z['biasCor_m_b']
    Z=Z[Z['zHEL']<0.8]
    l=Z.index.values.tolist() 
    l=np.array(l)
    N=len(Z) ; 

    if not options.DIP:
        name=name+"_FIX_DIP_TO_"
        if options.DIPDIR==1:
            radip=CMBdipra
            decdip=CMBdipdec
            name=name+"CMB_"
    else:
        name=name+"_FLOAT_DIP"
    Z=Z.to_numpy()
    ZINDEX=options.ZINDEX
    
    if ZINDEX==9:
        radip= 162.95389715
        decdip=-25.96734154
    
    dct={0:'zHD',7:'zHEL',8:'zCMB',9:'zLG'}

    def MUZ(Zc, Q0, J0,S0=None,L0=None):

        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   

        if np.any(np.isnan(k)):
                
            k[np.isnan(k)] = 63.15861331456834
        return k
    def dLPhenoF3(z, q0, j0):
        dl=z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.)*(1+Z[:,7])/(1+z)
        return dl
    COVd = np.load( '../statsys_mbcorr.npy' ) [:,l][l]# Constructing data covariance matrix w/ sys.
    def COV(sM=0,RV=0):
        COVl=np.diag((sM**2)*np.ones(N))
        if RV==0: 
            return np.array(COVl+COVd)
        elif RV==1:
            return np.array( COVd )
        elif RV==2:
            return np.array(COVl)

    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))

    def RESVF3( qm0, J0  , M0 ): 
        Y0 = np.array([M0])
        mu = MUZ(Z[:,ZINDEX], Q0, J0) ;
 
    
    def RESVF3Dip( qm0, qd0,J0 , M0,  DS=np.inf, stype = STYPE,ra=radip,dec=decdip): #Total residual, \hat Z - Y_0*A
        Y0A = np.array([ M0])
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        Zc = Z[:,ZINDEX]
        if stype=='NoScDep':
            Q = qm0 + qd0*cosangle
        elif stype=='Exp':
            Qdip = qd0*cosangle*np.exp(-1.*Zc/DS)
            Q = qm0 + Qdip
            if np.any(np.isnan(Q)  ) or np.any(np.isinf(Q)  ) or np.any(abs(Q)>1e12) :
                    print(Q)
                    Q[np.isnan(Q)] = 9960
                    Q[np.isinf(Q)] = 9960
                    Q[abs(Q)>1e12] = 9960
        mu = MUZ(Zc, Q, J0)
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )  

    def m2loglike(pars , RV = 0):
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')

        else:
            cov = COV( *[ pars[i] for i in [2] ] )
                
            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True )
            except np.linalg.linalg.LinAlgError: # If not positive definite
                return +13993*10.**20 
            except ValueError: # If contains infinity
                return 13995*10.**20

            res = RESVF3Dip(qm0,qd0, *[ pars[i] for i in [0,1,3] ] )
            

                    
      
            part_log = N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            if  pars[-1]<zlim :
                    part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))
            if RV==0:
                m2loglike = part_log + part_exp
                return m2loglike
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log 
 

    bnds = ((-10.,10.),(None,None),
                (None,None),(None,None),
                (None,None),(zlim,None))

    pre_found_best=np.array([ 5.97936846e-01, -1.93502269e+01 , 2.92985475e-02, 1.55757843e-02])

    MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12, options={'maxiter':194000})
    print(MLE)

    return MLE.fun, MLE.success
    
    


qm_ar=np.linspace(-0.65,0.1,5)  ## for ZHEL
qd_ar=np.linspace(-30,-1,5) ## for ZHEL

# qm_ar=np.linspace(-0.65,0.1,2)  ## for ZLG
# qd_ar=np.linspace(-40,1,2) ## for ZLG

# qm_ar=np.linspace(-0.65,0.1,50)  ## for ZLG
# qd_ar=np.linspace(-30,1,50) ## for ZLG 

x,y=np.meshgrid(qm_ar,qd_ar)
x1=np.reshape(x,(1,25))[0]
y1=np.reshape(y,(1,25))[0]
res=zip(x1,y1)
s=list(res)
print('qm:',qm_ar)
print('qd:',qd_ar)
results=[]

filename= f"/scratch/animesh.sah/NEWSCAN_QM_QD_ZINDEX_{options.ZINDEX}_v3"+".txt"
with open(filename,'w') as f:
    pass

with open(filename,'a') as f:
    print('qm:',qm_ar,file=f)
    print('qd:',qd_ar,file=f)

if options.PARALLEL:
    for arg1,arg2 in s:
        results.append(Minimizer.remote(arg1,arg2))
    res=ray.get(results)
    
    with open(filename,'a') as f:
        print(res,file=f)
    ray.shutdown()
    b=time.time()
    print("Time taken:",b-a)
    
else:
    lst=[]
    for arg1, arg2 in s:
	    m=Minimizer(arg1,arg2)
	    lst.append(m)
    print(lst)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:13:55 2024

@author: shin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-Dipole_LG_DIR_Powell
"""
Created on Mon Dec 18 21:18:47 2023

@author: shin
"""

import numpy as np
from scipy import interpolate, linalg, optimize ,integrate
from optparse import OptionParser
from collections import OrderedDict
import pickle
import time
import sys
import pandas as pd
from astropy.coordinates import angular_separation as ang 
import matplotlib.pyplot as plt
import astropy.units as u
usage = 'usage: %prog [options]'
parser = OptionParser(usage)
parser.add_option( "-d", "--details", action="store", type="int", default=2, dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option( "-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option( "-s", "--scan", action = "store_true", default=False, dest="SCAN", help = "Whether to do a scan")
parser.add_option( "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell. ")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole too?")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO?")
parser.add_option( "-e","--evaluate", action = "store",type='int', dest="EVAL", default=3, help = "What evaluation needs to be done, 1: LCDM , 2: z Taylor , 3: z Dipole Taylor , 4: H0 dipole, 5: H0,Q0 dipole ,6: Quadrupolar ?")

parser.add_option( "--zind", action = "store", type='int',default=9, dest="ZINDEX", help = "ZINDEX?")

parser.add_option( "--arg1", action = "store", type="float", default=0, dest="ARG1", help = "ARGUMENT1 for RA?")
parser.add_option( "--arg2", action = "store", type="float", default=0, dest="ARG2", help = "ARGUMENT2 for DEC? ")
parser.add_option("--zlim", action = "store",type="float", default=0.00937, dest="ZLIM")
(options, args) = parser.parse_args()


if options.DET==1:
    STYPE='NoScDep'
elif options.DET==2:
    STYPE='Exp'

else:
    STYPE='None'
    

if options.MET==1:
    met='Nelder-Mead'
elif options.MET==2:
    met='SLSQP'
elif options.MET==3:
    met='Powell'
elif options.MET==4:
    met="trust-constr"
elif options.MET==5:
    met="TNC"
elif options.MET ==6:
    met="COBYLA"
elif options.MET ==7:
    met="L-BFGS-B"
elif options.MET==8:
    met= "Newton-CG"
elif options.MET==9:
    met='BFGS'
elif options.MET==10:
    met='CG'
elif options.MET==11:
    met='trust-exact'

c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

CMBdipdec = -7
CMBdipra = 168
two_mppra=164.6877348*u.deg 
two_mppdec=-17.1115932*u.deg
raLG_SUN= 333.53277784*u.deg
decLG_SUN = 49.33111122*u.deg

vLG_SUN= 299 #km/s
two_mppra_outer=194.80253185*u.deg 
two_mppdec_outer=-56.85562848*u.deg
raBF_SUN=330.0724108*u.deg
decBF_SUN=-51.6812964*u.deg

path='/Storage/animesh/DATA_PPLUS'

def Minimizer(zlim1=0,ra0=None ,dec0=None):

    index=np.load('/Storage/animesh/Analysis_C2/index_sorted_lane.npy')
    Z = pd.read_csv(path+'/Zpan.csv')
    name=str(met)
    df=pd.read_csv( path+'/Pantheon+SH0ES.dat',delimiter=' ')
    if options.REVB:
        print ('reversing bias')
        Z['mB'] = Z['mB'] + df['biasCor_m_b']

    Z=Z.loc[index]
    Z=Z.reset_index(drop=True)
    Z=Z[Z['zHEL']<0.8]
    if zlim1!=0:
        print("Using the zlim provided",zlim1)
        Z=Z[Z['zHEL']>zlim1]
        
    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
    #zSUN=np.sqrt((1-vcmb*np.cos(ang(racmb,deccmb,ra,dec)).value/c)/(1+vcmb*np.cos(ang(racmb,deccmb,ra,dec)).value/c))-1
    zDF=np.sqrt((1-vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c)/(1+vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c))-1
    zLG=((1+Z['zHEL'])/(1+zDF))-1

    
    
    l=Z.index.values.tolist() 
    l=np.array(l)
    
    tempind=np.array([[3*i,3*i+1,3*i+2] for i in l])
    tempind=tempind.flatten(order='c')
    N= len(Z)
    Z['zLG']=zLG

    Z=Z.to_numpy()
    INDEX_DCT={9:"zHEL",0:"zHD",10:"zCMB",11:"ZLG",12:'zBF'}
    
    ZINDEX=options.ZINDEX
    print('Using Redshift:',INDEX_DCT[ZINDEX])  
    if not options.DIP:
        name=name+"_FIX_DIP_TO_"
        if options.DIPDIR==1:
            radip=CMBdipra
            decdip=CMBdipdec
            name=name+"CMB_"
    else:
        name=name+"_FLOAT_DIP"
    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))
        
      
    
    def MUZ(Zc, Q0, J0,S0=None,L0=None,OK=None):
        #print("Q0:",Q0,"J0:",J0,"S0:",S0,"L0:",L0)


        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    
    

    
    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.) *(1+Z[:,9])/(1+z)
    

    
    if ZINDEX==11:
        radip= 162.95389715
        decdip=-25.96734154

    name+=str(INDEX_DCT[ZINDEX])+'_'
    print("Using Redshift:",INDEX_DCT[ZINDEX])
    zlim=Z[:,ZINDEX][0]

    COVd = np.load('/Storage/animesh/Analysis_C2/cov_final.npy')
    #print(COVd.shape)
    COVd=COVd[:,tempind][tempind]

    
    def COV( sM=0.03,A=1.23055320e-01 ,sX=9.57326949e-01, B=2.13367616e+00, sC=7.73822769e-02 , RV=0): # Total covariance matrix


        block3 = np.array( [[sM**2 + (sX**2)*A**2 + (sC**2)*B**2,    -(sX**2)*A, (sC**2)*B],
                                                    [-(sX**2)*A , (sX**2), 0],
                                                    [ (sC**2)*B ,  0, (sC**2)]] )
        ATCOVlA = linalg.block_diag( *[ block3 for i in range(N) ] ) ;
        
        if RV==0:
            return np.array( COVd + ATCOVlA );
        elif RV==1:
            return np.array( COVd );
        elif RV==2:
            return np.array( ATCOVlA );
    

    
    


    
    def RESVF3Dip(M0,  A ,X0, B , C0,Q0,J0, QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        ra=ra0
        dec=dec0
        cosangle = cdAngle(ra, dec, Z[:,6], Z[:,7])
        Zc = Z[:,ZINDEX]
        
        #print("M0",M0,"A",A,"X0",X0,"B",B,"C0",C0,"Q0",Q0,"J0",J0,"QD",QD,"DS",DS)

        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            #print 'Here', Qdip
        Q = Q0 + Qdip
        mu = MUZ(Zc, Q, J0) ;
        #print 'Now', Q0, J0, K, V0, V1
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
     
 
    if options.EVAL==3:
        print("Evaluating dipole")
        function=RESVF3Dip
    elif options.EVAL==4:
        print("Evaluating dipole")
        function=RESVF3_H0Dip
    else:
        raise Exception("Invalid EVAL value")
        


    def m2loglike(pars , RV = 0):
        
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')
            
        else:  

            cov = COV( *[ pars[i] for i in [1,2,4,5,7] ] )
            
                

            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
            except np.linalg.linalg.LinAlgError: # If not positive definite
                print("LINALGERRROR error")
                return +13993*10.**20 
            except ValueError: # If contains infinity
                print("Value error")
                return 13995*10.**20

            
            res = function(*[pars[i] for i,val in enumerate(pars) if i!=1 and i!=4 and i!=7])

            
            part_log = 3*N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            

            if RV==0:
                m2loglike = part_log + part_exp
     

                return m2loglike 
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log 
    
    if options.EVAL==3:

        pre_found_best=np.array([-1.915e+01,  1.567e-01  ,1.583e-01, -5.836e-02 , 9.630e-01,
             3.175e+00, -3.845e-02,  5.522e-02 , 9.477e-03, -6.462e-01,
            -3.176e+01 , 9.380e-03])
        if options.ZINDEX==11:
            pre_found_best=[-1.918e+01 , 1.225e-01 , 1.755e-01, -7.126e-02,  9.659e-01,
                3.813e+00, -3.398e-02 , 5.824e-02, -2.774e-01,  4.638e-01,
                -5.793e+01 , 1.019e-02]
        elif options.ZINDEX==9:
            pre_found_best=[-1.916e+01 , 1.225e-01  ,1.753e-01, -7.080e-02,  9.658e-01,
             3.821e+00, -3.444e-02,  5.827e-02 ,-2.271e-01,  2.954e-01,
            -2.305e+01 , 1.107e-02]
        elif options.ZINDEX==0:
            pre_found_best=   [-1.918e+01 , 1.212e-01 , 1.747e-01, -7.063e-02,  9.661e-01,
             3.848e+00, -3.599e-02,  5.797e-02 ,-3.780e-01 , 9.560e-01,
             1.107e+01 , 1.044e-02]
        
        bounds=[(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(zlim,None)]
            

    else:
        print('Error: Wrong eval type')
        sys.exit(1)


    MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12 , options={'maxiter':19205000},bounds=bounds)


    return MLE.fun ,name, MLE.success ,MLE.x
                
 

   

ar=[0.00937]
filename=f'/Storage/animesh/Analysis_C2/Output/RA_DEC_SCAN_zIND={options.ZINDEX}.txt'        

    
i=options.ZLIM
ra=options.ARG1
dec=options.ARG2
print(ra,dec)
m=Minimizer(zlim1=i,ra0=ra,dec0=dec)
print(f'{options.ARG1},{options.ARG2},{m[0]},{m[-2]},{m[-1][-1]}')
with open(filename,'a') as f:
    print(f'{options.ARG1},{options.ARG2},{m[0]},{m[-2]},{m[-1][-1]}',file=f)
#print(m[0])



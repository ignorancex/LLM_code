#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:32:51 2023

@author: shin
"""


import numpy as np
from scipy import linalg ,optimize ,integrate
from optparse import OptionParser
import pandas as pd
import time
from astropy.coordinates import angular_separation as ang 
import matplotlib.pyplot as plt
import astropy.units as u
import sys
usage = 'usage: %prog [options]'
parser = OptionParser(usage) 

###note before using the code-

#use of option parser
# -e or --evaluate options sets what function inimization is to be done for exampe a lambda cdm or kinematic taylor expansion.
#-d or --details options sets the functional form of dipole
# -m or --method is the method used for minimization
# -r or --reversebias reverses bias corrections on the magnitude
# -t or --taylor remove supernovae above 0.8
# --dipoledir fixes dipole direction preferrred direction (right now it ony has one option i.e CMB dipole direction)
# --dipole can be used when you need to find dipole direction as well with the minimzed parameters
# --scanshelldip is used for estimating confidence intervals on paramters only in shell analysis (Use only with -e 7 or 8)
 
parser.add_option( "-e","--evaluate", action = "store", dest="EVAL", default=3, help = "What evaluation needs to be done, 1: LCDM , 2: z Taylor , 3: z Dipole Taylor , 4: H0 dipole, 5: H0 as well as Q0 dipole ?, 6: Qduadropolar analysis, 7: Shell analysis for deceleration parameter ,8: Shell anaysis for hubble parameter")
parser.add_option("-d", "--details", action="store", type="int", default=2,dest="DET", help=" 1: Fit for a non scale dependent dipolar modulation in Q.  2. Fit for an exponentially falling scale dependent dipolar modulation in Q. ")
parser.add_option("-m", "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell. ")
parser.add_option( "-z","--zind", action = "store",type="int", default=7, dest="ZINDEX", help = "Which Redshift to use : zHEL,ZCMB or ZHD? ")
parser.add_option("-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option("-t", "--taylor", action = "store_true", default=True, dest="TAY", help = "Remove high z redhsit for taylor analysis")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole too?")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO?")
parser.add_option("--scanshelldip", action = "store_true", default=False, dest="SCANSHELLDIPOLE")
parser.add_option("--fixpars", action = "store_true", default=False, dest="FIXPARS",help='Fixing parmater values for hubble parameter estimation? ')
# if fixpars is TRUE it fixes parameter values to following 

(options, args) = parser.parse_args()

dctdet={2:"Non scalar",3:"Flat",4:"Exponential",5:"Linear",6:'Power',8:'mix'}

if options.DET==1:
    STYPE='NoScDep'

elif options.DET==2:
    STYPE='Exp'

elif options.DET==3:
    STYPE='mix'
else:
    STYPE='None'
    
if not options.DIP:
    if options.DIPDIR==1:
        print("Dipole Direction is CMB")
    elif options.DIPDIR==2:
        print("Dipole Direction is LG-SUN")

else:
    print("Floating Dipole")

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

print("Using Method: ",met,"\nUsing profile: ",dctdet[options.DET] )

shell_width=100



if options.EVAL==1:
    evalu='lkly'
elif options.EVAL==2:
    evalu='RESVF3'
elif options.EVAL==3:
    evalu='RESVF3Dip'
elif options.EVAL==4:
    evalu="RESVF3_H0Dip"
elif options.EVAL==5:
    evalu="RESVF3Q0_H0dip"
elif options.EVAL==6:
    evalu="RESVF3Quad"
c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

CMBdipdec = -7
CMBdipra = 168


LGra= 164.6877348

LGdec= -17.1115932
LGvel=620 #km/s
vcmb=369
racmb=168
deccmb=-7

racmb=168
deccmb=-7

raSUN_LG= 333.53277784*u.deg
decSUN_LG = 49.33111122*u.deg
vLG_SUN= 299 #km/s


raLG_SUN= 153.53277784*u.deg
decLG_SUN = -49.33111122*u.deg


H0=70





if  options.EVAL>6 and options.DET!=1:
    print('Error: need to give correction functional form for shell analysis')
    sys.exit(1)
    
def Minimizer(init=None,final=None,zlim1=0,dip=None):
    radip=0
    decdip=0
    Z = pd.read_csv( 'Z_mbcorr.csv' )
    df=pd.read_csv('Pantheon+SH0ES.dat',delimiter=' ')
    name=str(met)
    if options.REVB:
        print ('reversing bias')
        Z['m_b_corr'] = Z['m_b_corr'] + df['biasCor_m_b']
    Z=Z.sort_values('zHEL')
    

    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
    zDF1=np.sqrt((1-vLG_SUN*np.cos(ang(raSUN_LG,decSUN_LG,ra,dec)).value/c)/(1+vLG_SUN*np.cos(ang(raSUN_LG,decSUN_LG,ra,dec)).value/c))-1

    zLG=((1+Z['zHEL'])/(1+zDF1))-1           ##using redshift addition


    Z['zLG']=zLG

  
    
    if options.TAY:
        print("Removing higher redshifts")
        Z=Z[Z['zHEL']<0.8]
        name+='TAYLOR_EXP_'
 
    l=Z.index.values.tolist() 
    l=np.array(l)
  

    status=" "

    if zlim1!=0:
        status="Using zlim_"+str(zlim1)
        print("Using the zlim provided",zlim1)
        l=l[Z['zHEL']>zlim1]
        Z=Z[Z['zHEL']>zlim1]
        
    elif init!=None and final!=None:
        status="Using shell_"+str(init/100)
        print("Using the shell provided:",init,final)
        l=l[init:final]
        Z=Z.iloc[init:final]
        


   
       
    med= np.median(Z['zHEL'])
    Z=Z.to_numpy()
    ZINDEX=options.ZINDEX
    print(len(Z))
    

    zlim=Z[0,ZINDEX]
    dct={0:'zHD',7:'zHEL',8:'zCMB',9:'zLG',10:'zBF'}
    print("USING redshift: ",dct[ZINDEX] )
    N=Z.shape[0]
    name=name+"Z="+str(dct[ZINDEX])

    if not options.DIP:
        name=name+"_FIX_DIP_TO_"
        if options.DIPDIR==1:
            
            radip=CMBdipra
            decdip=CMBdipdec
            
            name=name+"CMB_"

    else:
        name=name+"_FLOAT_DIP"
    
 
    if ZINDEX==9:
        radip= 162.95389715
        decdip=-25.96734154   # values taken from Planck 2018 paper
    ## for LAMBDA CDM analysis

    def func(Zc,OM,OL,zh=None,zp=None):
            OK= 1.-OM-OL
            def I (z):
                return 1./np.sqrt(OM*(1+z)**3+OL+OK*(1+z)**2)
            if OK==0:
                integ=integrate.quad(I,0,Zc)[0]
            elif OK>0:
                integ= (1./OK)**0.5 *np.sinh(integrate.quad(I,0,Zc)[0]*OK**(0.5))
            elif OK<0:
                integ= (-1./OK)**0.5 *np.sin(integrate.quad(I,0,Zc)[0]*(-OK)**(0.5))
            if zp is not None:
                return (1.+zp)*(1+zh)*integ
            elif zh is not None:
                return (1.+zh)*integ
                            
            return (1.+Zc)*integ
    def dL_lcdm(Zc, OM, OL, Zh=None, Zp=None):
        if Zp is not None:
            return np.hstack([func(zc, OM, OL, zh, zp) for zc, zh, zp in zip(Zc, Zh, Zp)])
        elif Zh is not None:
            return np.hstack([func(zc, OM, OL, zh) for zc, zh in zip(Zc, Zh)])
        return np.hstack([func(z, OM, OL) for z in Zc])
    def MU_lcdm(Zc, OM, OL):
        if options.LCDM:
            k = 25 +5*np.log10((c/H0 )*dL_lcdm(Zc,OM,OL))
                
        if np.any(np.isnan(k)):
            print ('Fuck', OM, OL)
            k[np.isnan(k)] = 63.15861331456834
        return k

    def MUZ(Zc, Q0, J0,S0=None,OK=None,L0=None): 
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
        if np.any(np.isnan(k)):
                
            k[np.isnan(k)] = 63.15861331456834
        return k
    def MUZ_H0Dip(Zc, Q0, J0,H0):
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.  
        if np.any(np.isnan(k)):
                
            k[np.isnan(k)] = 63.15861331456834
        return k
    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.)*(1+Z[:,7])/(1+z)
    
    def dLPhenoF4( z,q0, j0,s0,OK):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.+
                         1/24*(2-2*q0-15*q0**2-15*q0**3+5*j0+10*q0*j0+s0 )*z**3)
   
    def dLPhenoF4_OK( z,q0, j0,s0,OK):
        
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0-OK)*z**2.+
                         1/24*(2-2*q0-15*q0**2-15*q0**3+5*j0+10*q0*j0+s0-2*OK*(1+3*q0) )*z**3)
    def dLPhenoF5( z,q0, j0,s0,l0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.+
                         1/24*(2-2*q0-15*q0**2-15*q0**3+5*j0+10*q0*j0+s0 )*z**3
                        +(1/120)*(-6+6*q0+81*q0**2+165*q0**3+105*q0**4+10*j0**2-27*j0-110*q0*j0-105*j0*q0**2-15*q0*s0-11*s0-l0)*z**4)
    

    COVd = np.load( 'statsys_mbcorr.npy' ) [:,l][l]# Constructing data covariance matrix w/ sys.
    
    def COV(sM=0.045,RV=0):
        #sM=0.045
        #sM=9.495e-02
        #sM=8.531e-02
        #sM=1e-8
        #sM=8.531e-02
        
        if options.EVAL>6:    #fixes the parameters 
            if ZINDEX==7:
                sM=8.531e-02
            elif ZINDEX==8:
                sM=5.776e-02
            elif ZINDEX==9:
                sM=9.495e-02
        COVl=np.diag((sM**2)*np.ones(N))
     
        if RV==0: 
            return np.array(COVl+COVd)
            
        elif RV==1:
            return np.array( COVd )
        elif RV==2:        
            return np.array(COVl)
        
    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))
        
    def RESVF3(M0, Q0, J0,S0=None,L0=None  ): #Total residual, \hat Z - Y_0
        #print("Q0:",Q0,"J0:",J0,"M0:",M0,"S0",S0,"L0",L0)
        #M0=-19.3
        
        Y0 = np.array([M0])
        #mu=MUZ(Z[:,ZINDEX], Q0,J0);
        mu = MUZ(Z[:,ZINDEX], Q0, J0,S0,L0);
        return  np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0 ) for i in range(N) ] ) 
    def RESVF3_H0Dip(Hm, Q0, J0 ,Hd,ra=radip,dec=decdip,DS=np.inf,stype = STYPE):#Total residual, \hat Z - Y_0
        #Hd=0
# =============================================================================
#         if ZINDEX==7:
#             Q0=-4.885e-02 
#             J0=-9.390e-01
#             
#         if ZINDEX==8:
#             #pass
#             Q0=-2.414e-01 
#             J0=-3.285e-01
#         if ZINDEX==9:
#             Q0=-3.561e-02 
#             J0=-1.039e+00
#         if ZINDEX==0:
#             Q0=-3.331e-01
#             J0=-1.787e-02 
# =============================================================================
            
            
            
        #Q0=-0.55
        #J0=1
        Zc = Z[:,ZINDEX]
        M0=-19.25

    
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        
        if stype=='NoScDep':
            H0 = Hm + Hd*cosangle
        elif stype=='Exp':
             Hdip =Hd*cosangle*np.exp(-1.*Zc/DS)
             H0 = Hm  + Hdip
        else:
            print("WRONG STYPE")
            sys.exit(1)
        Y0A = np.array([ M0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q0, J0,H0) ;
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )  
    
     
    
    
    def lkly( M0,OM,OL ):
        Y0 = np.array([ M0])
        mu = MU_lcdm(Z[:,ZINDEX], OM,OL) ;
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0 ) for i in range(N) ] )  
    
    def RESVF3Quad(Hm,Q0,J0,QD,DS,DS1,lam1,lam2):
        print(f'Hm{Hm},QD:{QD},J0:{J0},lam1:{lam1},lam2:{lam2},DS:{DS},DS1:{DS1}')
        M0=-19.25
        Zc = Z[:,ZINDEX]
        lam1=0
        lam2=0
        cosangle1 = cdAngle(193.36,32.11, Z[:,5], Z[:,6])**2
        cosangle2= cdAngle(248.52,-41.85, Z[:,5], Z[:,6])**2
        cosangle3 = cdAngle(306.43,31.04, Z[:,5], Z[:,6])**2
        cosangle = cdAngle(167.94,-6.94, Z[:,5], Z[:,6])
        DS1=0.1/np.log(2)
        
        H0 = Hm*(1+(lam1*cosangle1+lam2*cosangle2-(lam1+lam2)*cosangle3)*np.exp(-Zc/DS1))
        
        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
        Q = Q0 + Qdip
        Y0A = np.array([ M0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q, J0,H0) ;
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )  
        
        
        
        


                
    def RESVF3Dip( M0,Q0, J0 , QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        
    
        #print("qd",QD,"QM",Q0,"JO",J0,"m0",M0,"S",DS,'ra',ra,'dec',dec)
        #print('ra',ra,'dec',dec)
        #M0=-19.3
        #Q0=-4.163e-01
        #QD=0
# =============================================================================
        #QD=-6.041e-02
        #Q0=-0.4
# =============================================================================
        Y0A = np.array([ M0])
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        Zc = Z[:,ZINDEX]
        if stype=='NoScDep':
            Q = Q0 + QD*cosangle
        elif stype=='Flat':
            Qdip = QD*cosangle
            
            Qdip[Zc>(DS+0.1)] = 0
            Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            Q = Q0 + Qdip
        elif stype=='Exp':

            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip

        elif stype=='Lin':
            Qd = QD - Zc*DS
            Qd[Qd<0] = 0
            Q = Q0 + Qd*cosangle
        elif stype=='Power':
            Qd = QD*cosangle/(1+Zc)
            Q = Q0 + Qd*cosangle
            
        mu = MUZ(Zc, Q, J0) ;
        #print(mu)  
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )
    
    def RESVF3Q0_H0dip( M0,Q0, J0 , QD, Hd,DS=np.inf,DS1=np.inf, stype = STYPE): #Total residual, \hat Z - Y_0*A
        #ra2=ra1=raLG_SUN.value
        #dec2=dec1=decLG_SUN.value
        ra1=ra2=racmb
        dec1=dec2=deccmb
        print("qd",QD,"QM",Q0,'Hd',Hd,"JO",J0,"M0",M0,"S1",DS,'S2',DS1,'ra1',ra1,'dec1',dec1,'ra2',ra2,'dec2',dec2)
        #print('ra',ra,'dec',dec)
        
        
        #use this if you want to fix some parmeters
        '''
        #M0=-1.93391147e+01
        Q0=-4.23205928e-01
        #Q0=-0.19
        J0=4.29864806e-01
        #Qd = -1.762e+01
        #Qd = -9.4
        DS= 9.380e-03
        #Q0=-0.369
        #Hd=0
        #J0=0.1210.56
        '''
        
        Hm=70
        Y0A = np.array([ M0])
        cosangle1 = cdAngle(ra1,dec1, Z[:,5], Z[:,6])
        cosangle2 = cdAngle(ra2,dec2, Z[:,5], Z[:,6])
        Zc = Z[:,ZINDEX]
        if stype=='mix':
            Qdip = QD*cosangle1*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip
            H0 = Hm + Hd*cosangle2


        if stype=='NoScDep':
            Q = Q0 + QD*cosangle1
            H0 = Hm + Hd*cosangle2
        
        elif stype=='Exp':

            Qdip = QD*cosangle1*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip
            #Q=Q0
            Hdip = Hd*cosangle2*np.exp(-1.*Zc/DS1)
            #Hdip = -1.732*cosangle2*np.exp(-1.*Zc/0.496)

            H0 = Hm + Hdip
           
   
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q, J0,H0) ; 
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )
    
    def RESVF3DipF4( M0,Q0, J0 ,S0,OK, QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        print("qd",QD,"QM",Q0,"JO",J0,"m0",M0,"S",DS,'S0',S0,'OK',OK)
        #print('ra',ra,'dec',dec)
        Y0A = np.array([ M0])
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        Zc = Z[:,ZINDEX]
        if stype=='NoScDep':
            Q = Q0 + QD*cosangle
        elif stype=='Flat':
            Qdip = QD*cosangle
            
            Qdip[Zc>(DS+0.1)] = 0
            Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            Q = Q0 + Qdip
        elif stype=='Exp':

            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip

        elif stype=='Lin':
            Qd = QD - Zc*DS
            Qd[Qd<0] = 0
            Q = Q0 + Qd*cosangle
        
            
        mu = MUZ(Zc, Q, J0,S0,OK) ;
           
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )
    def RESVF3DipF5( M0,Q0, J0 ,S0,L0, QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        #print("qd",QD,"QM",Q0,"JO",J0,"m0",M0,"S",DS,'S0',S0,'L0',L0)
        #print('ra',ra,'dec',dec)
        Y0A = np.array([ M0])
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        print(ra,dec)
        print(Z[:,5],Z[:,6])
        print(cosangle)
        
        Zc = Z[:,ZINDEX]
        if stype=='NoScDep':
            Q = Q0 + QD*cosangle
        elif stype=='Flat':
            Qdip = QD*cosangle
            
            Qdip[Zc>(DS+0.1)] = 0
            Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            Q = Q0 + Qdip
        elif stype=='Exp':

            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip

        elif stype=='Lin':
            Qd = QD - Zc*DS
            Qd[Qd<0] = 0
            Q = Q0 + Qd*cosangle
            
        mu = MUZ(Zc, Q, J0,S0,L0) ;
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )
    
    
    def RESVF3_H0Dip_noscdep(  Hd,M0,ra= radip,dec=decdip,Q0=-4.885e-02,J0=-9.390e-01,DS=np.inf,stype = STYPE):#Total residual, \hat Z - Y_0
        
        if ZINDEX==8:
            #Q0=-3.973e-01  
            #J0=2.522e-01
            
            #these values are from global fit 
            Q0=-2.414e-01 
            J0=-3.285e-01
        if ZINDEX==9:
            
            #these values are from global fit 
            qm=-3.561e-02 
            J0=-1.039e+00
            
            
            #qm=-0.326 
            #J0=-0.073 
            #qm=-0.217
            #J0= -0.356
            #qm=-0.247
            #J0=-0.303
            #qm=-0.55
            #J0=1
        
        if ZINDEX==0:

            #these values are from global fit 
            qm=-3.331e-01
            J0=-1.787e-02 
            
            
            #Q0=-0.3412
            #J0=0.0179

       
        #M0=-19.30
        Hm=70
        #M0=-19.30
        Zc = Z[:,ZINDEX]
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        
        if stype=='NoScDep':
            H0 = Hm + Hd*cosangle
        elif stype=='Exp':
             Hdip =Hd*cosangle*np.exp(-1.*Zc/DS)
             H0 = Hm  + Hdip
        else:
            print("WRONG STYPE")
            sys.exit()
        Y0A = np.array([ M0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q0, J0,H0) ;
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )  
        
    def RESVF3_noscdep(  qd,M0,ra= radip,dec=decdip,qm=-4.885e-02 ,J0=-9.390e-01,DS=np.inf,stype = STYPE):#Total residual, \hat Z - Y_0
        
        if ZINDEX==8:
            qm=-2.414e-01 
            J0=-3.285e-01
        if ZINDEX==9:
            qm=-3.561e-02 
            J0=-1.039e+00
        if ZINDEX==0:
            qm=-3.331e-01
            J0=-1.787e-02 
        
        Zc = Z[:,ZINDEX]
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        if stype=='NoScDep':
            q0 = qm + qd*cosangle
        else:
            print("WRONG STYPE")
        Y0A = np.array([ M0 ])
        mu = MUZ(Z[:,ZINDEX], q0, J0) 
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )  
            
    if options.EVAL==1:
        function=lkly
    elif options.EVAL==2:
        function=RESVF3
    elif options.EVAL==3:
        print("Evaluating dipole")
      

        function=RESVF3Dip
    elif options.EVAL==4:
        function=RESVF3_H0Dip
    elif options.EVAL==5:
        function=RESVF3Q0_H0dip
    elif options.EVAL==6:
        function=RESVF3Quad
    elif options.EVAL==7 :
        function=RESVF3_noscdep
    elif options.EVAL==8:
        function=RESVF3_H0Dip_noscdep
    else:
        print('Error: Wrong eval value')
        sys.exit(1)
    
    name+='_'+str(function)  
    def m2loglike(pars , RV = 0):
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')
        else:
            if options.EVAL<=6:
                 cov = COV( *[ pars[i] for i in [1] ] )
                 #cov=COV()
            else:
                 cov = COV()
                 #cov = COV( *[ pars[i] for i in [6] ] )

                 
            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True )
            except np.linalg.linalg.LinAlgError: # If not positive definite
                return +13993*10.**20 
            except ValueError: # If contains infinity
                return 13995*10.**20
            
            #print(function)
            #print('pars',*[pars[i] for i,val in enumerate(pars) if i!=1])
            if  options.EVAL<=6:
                res=function(*[pars[i] for i,val in enumerate(pars) if i!=1])
            else:
                #res = function(*[pars[i] for i,val in enumerate(pars) if i!=1 ])
                res = function(*[pars[i] for i,val in enumerate(pars) if i!=6 ])

            part_log = N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            
            if RV==0:
                m2loglike = part_log + part_exp
                #print(pars, m2loglike)
                return m2loglike 
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log  

   

    #pre_found_best=np.hstack([pre_found_best,ar])
    #pre_found_best=np.array([-19.32,0.03,-0.55,1,-0.35,3.11])\
    pre_found_best=[-1.93145269e+01 ,0.03  ]
    #pre_found_best=[70 ,0.03  ]
    bounds=((-21,-16),(None,None))
    if options.EVAL==1:
        name=name+" LCDM"
        ar=[0.3,0.7]
        new=((None,None),(None,None))
        bounds+=((None,None),(None,None))
        pre_found_best=np.hstack([pre_found_best,ar])
    else: 
        ar=[ -0.35,0.12]
        bounds+=((None,None),(None,None))
        name=name+" TAY"
        pre_found_best=np.hstack([pre_found_best,ar])
 
        if options.EVAL==3:
            ar=[-6]
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET >1:
                name=name+" F=f(z,s)"
                ar=[0.02]
                bounds+=((zlim,None),)
                pre_found_best=np.hstack([pre_found_best,ar])
        elif options.EVAL==4:
            pre_found_best[0]=70
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            ar=[-2]
            name=name+" H0dip"
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if options.DET ==2:
                name=name+" F=f(z,S)"
                ar=[0.02]
                bounds+=((zlim,1),)
                pre_found_best=np.hstack([pre_found_best,ar])
        
        elif options.EVAL==5:
            ar=[-6,-6]
            bounds+=((None,None),(None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET==2:
                name=name+" F=f(z,s)"
                ar=[0.02]
                bounds+=((zlim,None),)
                pre_found_best=np.hstack([pre_found_best,ar])
                if options.DET==2:
                    ar=[0.02]
                    bounds+=((zlim,None),)
                    pre_found_best=np.hstack([pre_found_best,ar])
            if options.DIP:
                ar=[132,35]
                #ar=[20,42]
                name=name+" Estimated Dipole"
                bounds+=((0,360),(-90,90))
                pre_found_best=np.hstack([pre_found_best,ar])
            
                
        elif options.EVAL==6:
            pre_found_best[0]=71.02
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            ar=[-1.671]
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            
            name=name+" F=f(z,S)"
            ar=[0.07]
            bounds+=((zlim,1),)
            pre_found_best=np.hstack([pre_found_best,ar])
            ar=[0.02]
            bounds+=((zlim,20),)
            pre_found_best=np.hstack([pre_found_best,ar])
            ar=[ 0.0253 , -0.0078]
            bounds+=((None,None),(None,None))
            pre_found_best=np.hstack([pre_found_best,ar])
        
        if options.DIP:
            ar=[175,-18]
            #ar=[20,42]
            name=name+" Estimated Dipole"
            bounds+=((0,360),(-90,90))
            pre_found_best=np.hstack([pre_found_best,ar])
        #pre_found_best=[-1.934e+01,  1.548e-02, -4.163e-01,  3.138e-01, -6.041e-02,9.464e-01]
        #pre_found_best=np.array([-1.93095718e+01 , 5.06578998e-09,  1.20125959e-01, -7.39598561e-02,9.58836522e-01,  2.00711958e+00, -1.93111624e-02 , 7.68943705e-02,-1.92238507e-01, -3.86475938e-01, -3.40900046e-01] )        if options.FIXA:
        if options.EVAL==7:
            pre_found_best=[-1,-19.3]
            bounds=((None,None),(None,None))
        if options.EVAL==8:
            pre_found_best=[0,70]
            bounds=((None,None),(None,None))
  
 
    print(f'fixing to: {dip}')
    def No_dip(pars):
        return pars[0]#- dip
    def qm_constr(pars):
        return pars[2]- (-4.163e-01)
    #,constraints = ({'type':'eq', 'fun':No_dip})
    constraints=({'type':'eq','fun':No_dip})
    print(bounds)
    print(pre_found_best)
    
    if options.SCANSHELLDIPOLE:
       MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-14 , options={'maxiter':150000},bounds=bounds,constraints=({'type':'eq','fun':No_dip}))
    else:
        
          #pre_found_best= [-1.935e+01,  2.918e-02 ,-4.394e-01 , 4.485e-01, -8.484e+00, 2.781e-02 , 1.331e+02,  3.212e+01]
          MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-14 , options={'maxiter':150000},bounds=bounds)#,constraints=({'type':'eq','fun':qm_constr}))

          #MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-14 , options={'maxiter':150000},bounds=bounds)

        
    print(MLE)
    print(name)

    
    return MLE.x, name, MLE.fun ,med,MLE

Z = pd.read_csv( 'Z_mbcorr.csv' )
Z=Z.sort_values('zHEL')

if options.TAY:
    Z=Z[Z['zHEL']<0.8]
N=len(Z)




#ar= [ 0.00355, 0.00435, 0.00491, 0.00523, 0.00587, 0.00662,0.00719, 0.00759, 0.00856, 0.00907,0.00947,0.01057]

#ar=np.array([ 0.00587, 0.00907, 0.01351, 0.01613, 0.01826, 0.02121,0.02324, 0.02531, 0.02873, 0.03139, 0.03486, 0.04146, 0.05057,0.06976, 0.10762])



ar=[0.00937]


#for i in ar:
#    m=Minimizer(zlim1=i)
#    print(m)
Z = pd.read_csv( 'Z_mbcorr.csv' )
Z=Z.sort_values('zHEL')
Z=Z[Z['zHEL']<0.8]
N=len(Z)
median=[]
if options.EVAL>6: 
    
    #m=Minimizer(zlim1=0.023)
    #print(m)
    #print(hahaha)
    
    
    if not options.SCANSHELLDIPOLE:
       
        shell_width=100
        num_splits=int(np.ceil(N/shell_width))
        init=0
        ar=[]
        ar2=[]
        mle=[]
        for i in range(num_splits):
            final=shell_width*(i+1)
            l=Z.index.values
            m=Minimizer(init=init,final=final)
            print(m)
            median.append(m [-1])

            init=final
            ar2.append(m[0][0])
            ar.append(m[0][1])
            mle.append(m[2])
            if options.DIP:
                ar.append([m[0].x[2],m[0].x[3]])
            
                
        print(f'x_true={ar2}\n ',f'MLE_true={mle}')
        print(ar)
        #print(f'median={median}\n ')

        if options.DIP:
            
            print(ar)
    else:
         if options.MET!=4 :
            print("Error: Method not supported for scanning \n Use trust Constraint")
            sys.exit()
        
         minimized_vals=[-38.86159740622688, -17.75354867779298, -8.208273573258992, -4.939017455520776, -4.178077420293118, -2.0032140695565652, -1.615233331256633, -0.7150906154827132, -0.1989928683734748, -0.044427387791486014, -0.14111061623341775, -0.3974002660681623, -0.0460520876681514, -0.11531701124053621, -0.2090319705703124, -0.011274688511929136, -0.057884518346162525]
         MLE=[342.76973980211324, -57.65151182449563, -51.79681046385484, -87.71259784624718, -84.76411854459579, -97.83179766640306, -105.91089817848456, -107.31134428317338, -106.5501546363131, -85.36903374173403, -97.60385279482867, -102.87869020484587, -93.4912505041787, -109.57416129361042, -84.22649126879598, -63.06914507196426, -54.34585282946863]

         shell_width=100
         init=0
         ar=[]
         fun=[]
         num_splits=int(np.ceil(N/shell_width))
         init=0
         if options.EVAL==7:
             for i in range(num_splits):
                 final=shell_width*(i+1)
                 if i < 3:
                     dipole=list(np.linspace(minimized_vals[i]-9,minimized_vals[i]+9,30))
                 elif 3<=i<10:
                     dipole=list(np.linspace(minimized_vals[i]-2,minimized_vals[i]+2,30))
                 elif i >=10 :
                     dipole=list(np.linspace(minimized_vals[i]-0.5,minimized_vals[i]+0.5,30))
                     
                 
                 l=Z.index.values
                 ar2=[]
                 for dip in dipole:
                         m=Minimizer(init=init,final=final,dip=dip)
                         ar2.append(m[2])
                         a=1
                 init=final
                     
                     
                 ar.append(dipole)
                 fun.append(ar2)
                 
         else:
             for i in range(num_splits):
                 final=shell_width*(i+1)
                 if i < 17:
                     dipole=list(np.linspace(minimized_vals[i]-3,minimized_vals[i]+3,30))
                 elif 6<=i<10:
                     dipole=list(np.linspace(minimized_vals[i]-0.7,minimized_vals[i]+0.7,30))
                 else :
                     dipole=list(np.linspace(minimized_vals[i]-0.3,minimized_vals[i]+0.3,30))

             l=Z.index.values
             ar2=[]
             for dip in dipole:
                     m=Minimizer(init=init,final=final,dip=dip)
                     ar2.append(m[2])
                     a=1
             init=final
                 
                 
             ar.append(dipole)
             fun.append(ar2)
             
         print('MLE=',fun,';x_ar=',ar)
         
             
     
else:
    MLE=[]
    MLE2=[]
    zlim=[0.0175,0.01,0.005]
    zlim=[0.023]
    zlim=[0.00937]
    qm=[]
    qd=[]
    S=[]
    zlim=[0.00587, 0.00907, 0.01351, 0.01613 ,0.01826, 0.02121, 0.02324, 0.02531, 0.02873,
 0.03139, 0.03486, 0.04146, 0.05057, 0.06976, 0.10762]
    zlim=[0.00937]

    for i in zlim:
        m=Minimizer(zlim1=i)
        MLE.append(m[0])
        qm.append(m[0][2])
        qd.append(m[0][-2])
        S.append(m[0][-1])
        #cov_mat_new=m[-1].hess_inv.todense()
        MLE2.append(m[-1].fun)
        print(m)

    #df=pd.DataFrame({'zlim':zlim,'qm':qm,'qd':qd,'S':S,'MLE':MLE2})        








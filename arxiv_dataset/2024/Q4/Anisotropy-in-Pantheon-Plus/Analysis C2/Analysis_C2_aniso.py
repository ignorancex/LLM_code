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

parser.add_option("-d", "--details", action="store", type="int", default=1,dest="DET", help=" 1: Fit for a non scale dependent dipolar modulation in Q.  2. Fit for an exponentially falling scale dependent dipolar modulation in Q. ")
parser.add_option( "-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option( "-s", "--scan", action = "store_true", default=False, dest="SCAN", help = "Whether to do a scan")

parser.add_option( "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell. ")

parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole too?")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO?")
parser.add_option( "-e","--evaluate", action = "store",type='int', dest="EVAL", default=7, help = "What evaluation needs to be done, 1: LCDM , 2: z Taylor , 3: z Dipole Taylor , 4: H0 dipole, 5: H0,Q0 dipole ,6: Quadrupolar,  7: Shell analysis for deceleration parameter ,8: Shell anaysis for hubble parameter")
parser.add_option( "--arg1", action = "store", type="float", default=0, dest="ARG1", help = "ARGUMENT1?")
parser.add_option( "--arg2", action = "store", type="float", default=0, dest="ARG2", help = "ARGUMENT2?")
parser.add_option("--zlim", action = "store",type="float", default=0.00937, dest="ZLIM")
parser.add_option( "--zind", action = "store", type='int',default=9, dest="ZINDEX", help = "Which Redshift to use  9: zHEL 10: zCMB 0: zHD 11: zLG?")
parser.add_option("--scanshelldip", action="store_true",default=False, dest="SCANSHELLDIPOLE")



parser.add_option("--fixpars", action = "store_true", default=True, dest="FIXPARS",help='Fixing parmater values for hubble parameter estimation? ')
# if fixpars is TRUE it fixes parameter values to following 
parser.add_option("--aa", action = "store",type="float", default=0.15, dest="A0")
parser.add_option("--bb", action = "store",type="float", default=3, dest="B0")
parser.add_option("--xx", action = "store",type="float", default=-0.075, dest="X0")
parser.add_option("--sM", action = "store",type="float", default=0.2, dest="sM0")
parser.add_option("--sC", action = "store",type="float", default=0.055, dest="sC0")
parser.add_option("--sX", action = "store",type="float", default=0.965, dest="sX0")
parser.add_option("--cc", action = "store",type="float", default=-0.042, dest="C0")


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

if  options.EVAL>6 and options.DET!=1:
    print('Error: need to give correction functional form for shell analysis')
    sys.exit(1)
path='/Storage/animesh/DATA_PPLUS'
def Minimizer(init=None,final=None,zlim1=0,qd=None ,qm=None,dip=None):
    A0=options.A0
    B0=options.B0
    C00=options.C0
    X00=options.X0
    sM0=options.sM0
    sC0=options.sC0
    sX0=options.sX0

    radip=None
    decdip=None
    name=str(met)
    index=np.load('/Storage/animesh/Analysis_C2/index_sorted_lane.npy')
    Z = pd.read_csv(path+'/Zpan.csv')

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
        
    elif init!=None and final!=None:
        print("Using the shell provided:",init,final)
    
        Z=Z.iloc[init:final]
    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
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
        #print('OM:',OM,'OL:',OL)
        k = 25 +5*np.log10((c/H0 )*dL_lcdm(Zc,OM,OL))
            #np.save("MU_new",k)
        if np.any(np.isnan(k)):
                print ('Wierd values ', OM, OL)
                k[np.isnan(k)] = 63.15861331456834
        return k
    
    def lkly(M0,  A ,X0, B , C0,OM,OL ):
        print("OM:",OM,"OL:",OL,"M0:",M0)
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MU_lcdm(Z[:,ZINDEX], OM,OL) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
    def RES( OM, OL , A , B , M0, X0, C0 ): #Total residual, \hat Z - Y_0*A
        print("OM",OM,"OL",OL,"M0",M0,"X0",X0,"C0",C0) 
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MU_lcdm(Z[:,ZINDEX], OM,OL) ;
        
        
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
        
    def MUZ(Zc, Q0, J0,S0=None,L0=None,OK=None):

        
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    
    
    #phenomenological taylor series expansion for dL from Visser et al
    #filename=f'TESTING.txt'
    
    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.) *(1+Z[:,9])/(1+z)
    
    
    def MUZ_H0Dip(Zc, Q0, J0,H0):
        #print(H0)
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25. 
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    
    def RESVF3_H0Dip(  Hm,A, X0,B, C0,Q0, J0 , Hd,ra=radip,dec=decdip,stype = STYPE):#Total residual, \hat Z - Y_0
        M0=-19.25
        Zc = Z[:,ZINDEX]
        if options.FIXPARS:
            A=A0
            B=B0
            X0=X00
            C0=C00
            if ZINDEX == 10:
                Q0 = 1.274e-01 
                J0=-7.807e-01
                
            if ZINDEX == 0:
                Q0 = -1.623e-01
                J0 = -4.772e-01
            if ZINDEX == 11:
                Q0 = 4.471e-01 
                J0=-1.049e+00
            if ZINDEX==9:
                Q0=3.796e-01 
                J0=-1.007e+00
        #Q0=-0.55
        #J0=1
        cosangle = cdAngle(ra,dec, Z[:,6], Z[:,7])
        
        if stype=='NoScDep':
            H0 = Hm + Hd*cosangle
        elif stype=='Flat':
            Hdip = Hd*cosangle
            H0[Zc>(DS+0.1)] = 0
            H0[Zc>DS] = H0[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            H0 = Hm + Hdip
        elif stype=='Exp':
            
            Hdip = Hd*cosangle*np.exp(-1.*Zc/DS)
            H0 = Hm + Hdip
        #print('/n/n/n\n\n\n\nH0/n/:',H0)
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q0, J0,H0) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
   
    def RESVF3_noscdep(qd, M0, ra=radip, dec=decdip, Q0=3.796e-01 , J0=-1.007e+00, A=0.15, B=3.0, X0=-0.075, C0=-0.042, stype=STYPE):
        

        if options.FIXPARS:
            A=A0
            B=B0
            X0=X00
            C0=C00
            if ZINDEX == 10:
                Q0 = 1.274e-01 
                J0=-7.807e-01
                
            if ZINDEX == 0:
                Q0 = -1.623e-01
                J0 = -4.772e-01
            if ZINDEX == 11:
                Q0 = 4.471e-01 
                J0=-1.049e+00
            if ZINDEX==9:
                Q0=3.796e-01 
                J0=-1.007e+00
        else:
            print('Need to fix params')
            sys.exit()
        #Q0=-0.55
        #J0=1
        Zc = Z[:, ZINDEX]
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])
        if stype == 'NoScDep':
            Q = Q0 + qd*cosangle
        else:
            print("WRONG STYPE")
        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MUZ(Z[:, ZINDEX], Q, J0)
       
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    
    def RESVF3_H0Dip_noscdep(  Hd,Hm,ra=radip,dec=decdip,Q0=4.54194622e-02,J0=-1.0152,A=0.1,B=0,X0=0,C0=1,stype = STYPE):

        
        M0=-19.25
        if options.FIXPARS:
            A=A0
            B=B0
            X0=X00
            C0=C00
            if ZINDEX == 10:
                Q0 = 1.274e-01 
                J0=-7.807e-01
                
            if ZINDEX == 0:
                Q0 = -1.623e-01
                J0 = -4.772e-01
            if ZINDEX == 11:
                Q0 = 4.471e-01 
                J0=-1.049e+00
            if ZINDEX==9:
                Q0=3.796e-01 
                J0=-1.007e+00
        
        else:
            print('Need to fix params')
            sys.exit()
  
        Zc = Z[:,ZINDEX]
        cosangle = cdAngle(ra,dec, Z[:,6], Z[:,7])
        
        if stype=='NoScDep':
            H0 = Hm + Hd*cosangle
        else:
            print("WRONG STYPE")

        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q0, J0,H0) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
    
    
    
    
    #covmatcomponents = [ "cal", "model", "bias", "dust", "sigmalens", "nonia" ]
    
    
    if ZINDEX==11:
        radip= 162.95389715
        decdip=-25.96734154

    name+=str(INDEX_DCT[ZINDEX])+'_'
    print("Using Redshift:",INDEX_DCT[ZINDEX])
    zlim=Z[:,ZINDEX][0]
    

    COVd = np.load('/Storage/animesh/Analysis_C2/cov_final.npy')
    COVdog=COVd
    print(COVd.shape)

    COVd=COVd[:,tempind][tempind]

    
    def COV( sM=0.03,A=1.23055320e-01 ,sX=9.57326949e-01, B=2.13367616e+00, sC=7.73822769e-02 , RV=0): # Total covariance matrix
        

        if options.FIXPARS:
            sM=sM0
            sX=sX0
            sC=sC0
            A=A0
            B=B0
        
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
    

    
    def RESVF3( M0,  A ,X0, B , C0,Q0,J0,S0=None,L0=None ): #Total residual, \hat Z - Y_0*A
        print('M0,A,X0,B,C0,Q0,J0',M0,A,X0,B,C0,Q0,J0)
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MUZ(Z[:,ZINDEX], Q0, J0,S0,L0);
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
    
    
    
    def RESVF3Dip(M0,  A ,X0, B , C0,Q0,J0, QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A

        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        cosangle = cdAngle(ra, dec, Z[:,6], Z[:,7])
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

        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
    def RESVF3Dip_temp( ra,dec,Q0, J0 , A , B , M0, X0, C0, QD, DS=np.inf, stype = STYPE): #Total residual, \hat Z - Y_0*A
        print("Q0:",Q0,"J0:",J0,"M0:",M0,"QD:",QD,"C0:",C0,"A:",A,"B:",B,"X0",X0,"DS:",DS)
        print(ra,dec)
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        cosangle = cdAngle(ra, dec, Z[:,6], Z[:,7])
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
        mu = MUZ(Zc, Q, J0) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
        
    def RESVF3Quad(Hm, A ,X0, B , C0,Q0,J0,QD,DS,DS1,lam1,lam2):
        print(f'Hm{Hm},QD:{QD},J0:{J0},lam1:{lam1},lam2:{lam2},DS:{DS},DS1:{DS1}')

        M0=-19.25
        Zc = Z[:,ZINDEX]
        
        cosangle1 = cdAngle(193.36632591, 32.10882412, Z[:,6], Z[:,7])**2
        cosangle2= cdAngle(248.52,-41.85, Z[:,6], Z[:,7])**2
        cosangle3 = cdAngle(306.43,31.04, Z[:,6], Z[:,7])**2
        cosangle = cdAngle(167.94,-6.94, Z[:,6], Z[:,7])
        lam1=0
        lam2=0
        DS1=0.1/np.log(2)
        print(f'Hm{Hm},QD:{QD},J0:{J0},lam1:{lam1},lam2:{lam2},DS:{DS},DS1:{DS1}')

        
        H0 = Hm*(1+(lam1*cosangle1+lam2*cosangle2-(lam1+lam2)*cosangle3)*np.exp(-1.*Zc/DS1))
        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
        Q = Q0 + Qdip
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        mu = MUZ_H0Dip(Z[:,ZINDEX], Q, J0,H0) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ])  
         
    def RESVF3_q0_H0Dip(M0,  A ,X0, B , C0,Q0,J0, QD,HD, DS1=np.inf,DS2=np.inf,ra1=radip,dec1=decdip,ra2=radip,dec2=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        H0=70
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        cosangle1 = cdAngle(ra1, dec1, Z[:,6], Z[:,7])
        cosangle2 = cdAngle(ra2, dec2, Z[:,6], Z[:,7])
        Zc = Z[:,ZINDEX]
        if stype=='mix':
            Qdip = QD*cosangle1*np.exp(-1.*Zc/DS1)
            Q = Q0 + Qdip
            Q = Q0 + Qdip 
            Q = Q0 
            Hd = H0 +HD*cosangle2*np.exp(-1.*Zc/DS2)
            
        if stype=='NoScDep':
            Q = Q0 + QD*cosangle1
            Hd = H0 +HD*cosangle2
        elif stype=='Flat':
            Qdip = QD*cosangle
            Qdip[Zc>(DS+0.1)] = 0
            Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            Q = Q0 + Qdip
        elif stype=='Exp':
            Qdip = QD*cosangle1*np.exp(-1.*Zc/DS1)
            Q = Q0 + Qdip
            Hdip = HD*cosangle2*np.exp(-1.*Zc/DS2)
            Hd = H0 + Hdip
     
        
        mu = MUZ_H0Dip(Zc, Q, J0,Hd) ;
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
    
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
        function=RESVF3_q0_H0Dip
    elif options.EVAL==6:
        function=RESVF3Quad

    elif options.EVAL==7 :
        function=RESVF3_noscdep
    elif options.EVAL==8:
        function=RESVF3_H0Dip_noscdep
    else:
        print('Error: Wrong eval value')
        sys.exit(1)


    def m2loglike(pars , RV = 0):
        
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')
            
        else:  
            if options.EVAL<=6:
                cov = COV( *[ pars[i] for i in [1,2,4,5,7] ] )
            else:
                cov = COV()
                
                
                

            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
            except np.linalg.linalg.LinAlgError: # If not positive definite
                print("LINALGERRROR error")
                return +13993*10.**20 
            except ValueError: # If contains infinity
                print("Value error")
                return 13995*10.**20

            if options.EVAL<=6:
                res = function(*[pars[i] for i,val in enumerate(pars) if i!=1 and i!=4 and i!=7])
            else:
                #print('EVAL>6')
                res = function(*[pars[i] for i,val in enumerate(pars) if i!=10 and i!=11 and i!=12] )

            part_log = 3*N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            

                    
            if RV==0:
                m2loglike = part_log + part_exp
                return m2loglike 
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log 
    


    

    pre_found_best=[-1.91875803e+01 , 1.21004511e-01  ,1.75236907e-01, -7.22271411e-02,
  9.66096801e-01 , 3.84331326e+00, -3.40308098e-02,  5.81056022e-02]
    bounds=((None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))
    if options.EVAL==1:
        name=name+" LCDM"
        ar=[0.3,0.7]
        new=((None,None),(None,None))
        bounds+=((None,None),(None,None))
        pre_found_best=np.hstack([pre_found_best,ar])
    else: 
        ar=[-0.20542328 ,  0.26429236]
        bounds+=((None,None),(None,None))
        name=name+" TAY"
        pre_found_best=np.hstack([pre_found_best,ar])
        
        if options.EVAL==3:
            ar=[-14]
            if ZINDEX==10:
                ar=[10]
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET ==2:
                name=name+" F=f(z,s)"
                ar=[zlim+0.01]
                bounds+=((zlim,1),)
                pre_found_best=np.hstack([pre_found_best,ar])
        elif options.EVAL==4:
            pre_found_best[0]=70
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            ar=[-3.40900046e-01]
            name=name+" H0dip"
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if options.DET ==2:
                name=name+" F=f(z,S)"
                ar=[0.02]
                bounds+=((zlim,1),)
                pre_found_best=np.hstack([pre_found_best,ar])
        elif options.EVAL==5:
            name+='QD_HD_EVAL_'
            ar=[-3,-9]
            bounds+=((None,None),(None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET >=1:
                name=name+" _F=f(z,s)_"
                ar=[0.009]
                bounds+=((zlim,None),)
                #bounds+=((None,None),)

                pre_found_best=np.hstack([pre_found_best,ar])
                if options.DET==2:
                    ar=[0.02]
                    bounds+=((zlim,None),)
                    #bounds+=((None,None),)

                    pre_found_best=np.hstack([pre_found_best,ar])
                if options.DIP:
                    ar=[2.34839835e+01,  3.80090097e+01]
                    bounds+=((0,360),(-90,90))
                    pre_found_best=np.hstack([pre_found_best,ar])
        elif options.EVAL==6:
            ar=[-14]
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            ar=[0.02,0.02,0,0]
            bounds+=((zlim,1),(zlim,1),(None,None),(None,None))
            pre_found_best[0]=70
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            pre_found_best=np.hstack([pre_found_best,ar])
            
            
        if options.DIP:
            ar=[133.4839835,  3.80090097e+01]
            if ZINDEX==11:
               ar=[1.44154014e+02, -1.10001955e+01]
            
            name=name+" Estimated Dipole"
            bounds+=((0,360),(-90,90))
            pre_found_best=np.hstack([pre_found_best,ar])
        

        
        if options.EVAL==7:
            pre_found_best=[-1,-19.3]
            bounds=((None,None),(None,None))
        if options.EVAL==8:
            pre_found_best=[0,70]
            bounds=((None,None),(None,None))
        print('length',len(pre_found_best))
        def No_dip(pars):
            return pars[0]-dip
        

        
        
    
    if not options.SCANSHELLDIPOLE:   
        MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12 , options={'maxiter':99205000},bounds=bounds)
    else:
        MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12 , options={'maxiter':1950000},bounds=bounds,constraints=({'type':'eq','fun':No_dip}))
        

    return MLE ,name 
                
      
if options.EVAL>6:   
    #m=Minimizer(zlim1=0.023)

    #print(haha)
    Z = pd.read_csv(path+'/Zpan.csv')
    
    Z=Z.sort_values('zHEL')
    Z=Z[Z['zHEL']<0.8]
    N=len(Z)
    shell_width=100
    #m=Minimizer(zlim1=0.0023)
    
    if not options.SCANSHELLDIPOLE:
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
            init=final
            ar2.append(m[0].x[0])
            ar.append([m[0].x[-2],m[0].x[-1]])
            mle.append(m[0].fun)
        print(f'No_sc_dep_dipole={ar2}')
        #print(ar)
        print(f'MLE_c2={mle}')
        print(ar)
        # Results already found
    else:
        if options.MET!=4 :
            print("Error: Method not supported for scanning \n Use trust Constraint")
            sys.exit()
        shell_width = 100
        ar = []
        fun = []
        
        num_splits = int(np.ceil(N/shell_width))
        init = 0
        x_true=[5.145070391343882, 4.74054922568041, 0.1421131262054647, 1.3351991877991178, 0.4262636415409181, 0.834373061922369, -0.32427117218981444, 0.010507648175100599, -0.16615594321350236, 0.0917091713301366, -0.2010308003198331, -0.12428593433167634, -0.12060829452892172, 0.013860950053726201, -0.13061229290380963, 0.04436252583713372, -0.04266619381074778]
        MLE_true=[272.2662419229142, 136.8774955683238, 121.70179710765206, 68.0022674756948, 55.09127999424521, 81.08761959986464, 7.880916617944251, -46.69719050167066, -54.637326585811394, -28.10294066578558, -28.079728604071477, -23.406030644590004, -29.14229851545423, -59.51726521767043, 13.234345736285775, -14.213504086941299, -13.049165132152211]


        minimized_vals = x_true
        if options.EVAL==7:
            for i in range(num_splits):
                final = shell_width*(i+1)
                if i < 3:
                    dipole = list(np.linspace(
                        minimized_vals[i]-9, minimized_vals[i]+9, 30))
                elif 3 <= i < 10:
                    dipole = list(np.linspace(
                        minimized_vals[i]-2, minimized_vals[i]+2, 30))
                elif i >= 10:
                    dipole = list(np.linspace(
                        minimized_vals[i]-2, minimized_vals[i]+2, 30))

                l = Z.index.values
                ar2 = []
                for dip in dipole:
                    m = Minimizer(init=init, final=final, dip=dip)
                    ar2.append(m[0].fun)
                    a = 1
                init = final

                ar.append(dipole)
                fun.append(ar2)

        else:
            print(num_splits)
            for i in range(num_splits):
                final = shell_width*(i+1)
                if i < 17:
                    dipole = list(np.linspace(
                        minimized_vals[i]-3, minimized_vals[i]+3, 20))
                elif 6 <= i < 10:
                    dipole = list(np.linspace(
                        minimized_vals[i]-0.7, minimized_vals[i]+0.7, 20))
                else:
                    dipole = list(np.linspace(
                        minimized_vals[i]-0.3, minimized_vals[i]+0.3, 20))
    
                l = Z.index.values
                ar2 = []
                for dip in dipole:
                    m = Minimizer(init=init, final=final, dip=dip)
                    ar2.append(m[0].fun)
                    a = 1
                init = final
    
                ar.append(dipole)
                fun.append(ar2)

        print('MLE=', fun, ';x_ar=', ar)
else:

        lst=[]
        
        
        ar=[0.00937]
        
        ar=[0.0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5]
        # 
        
        ar=[0,0.005,0.01,0.0175,0.025,0.0375,0.05,0.1]
    
        i=options.ZLIM


        m=Minimizer(zlim1=i)
        print(i,m)

        print(m[0].x)

        

        
        print('ZIND='+str(options.ZINDEX)+'_DET='+str(options.DET)+'_EVAL='+str(options.EVAL)+'_DIP='+str(options.DIP)+'zLIM='+str(options.ZLIM)+'revb='+str(options.REVB)+'met='+str(options.MET))





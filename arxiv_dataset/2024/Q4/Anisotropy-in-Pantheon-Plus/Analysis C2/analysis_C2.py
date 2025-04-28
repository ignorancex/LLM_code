#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:41:44 2023

@author: shin
"""
import sys
import numpy as np
from scipy import interpolate, linalg, optimize, integrate
from optparse import OptionParser
from collections import OrderedDict
import pickle
import time
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
# --fixastro fixes the astrophysical parameters for when shell analysis is to be done 
# -t or --taylor remove supernovae above 0.8
# --dipoledir fixes dipole direction preferrred direction (right now it ony has one option i.e CMB dipole direction)
# --dipole can be used when you need to find dipole direction as well with the minimzed parameters
# --scanshelldip is used for estimating confidence intervals on paramters only in shell analysis 


parser.add_option( "-e","--evaluate", action = "store", dest="EVAL", default=3, help = "What evaluation needs to be done, 1: LCDM , 2: Kinematic Taylor, 3:  Dipole in deceleration parameter Kinematic  , 4:  Dipole in Hubble parameter Kinematic, 5: H0 as well as Q0 dipole ?, 6: Qduadropolar Hubble analysis?")
parser.add_option("-d", "--details", action="store", type="int", default=4,dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option("-m", "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell.4: Trust-Constraint 5: TNC 6: Cobyla 7: L-BFGS-B 8: Newton-CG 9: BFGS 10: CG 11:trust-exact")
parser.add_option( "-z","--redshift", action = "store",type="int", default=7, dest="ZINDEX", help = "Which Redshift to use  9: zHEL 10: zCMB 0: zHD 11: zLG? ")
parser.add_option("-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")

parser.add_option( "--fixastro", action = "store", dest="FIXA", default=0, help = " 0: Doesnt fix astrophysical paramete,( Use only for shell analysis ) fix astrophysical paramters for 1: deceleration paramter 2: hubble parameter?")
parser.add_option("-t", "--taylor", action = "store_true", default=True, dest="TAY", help = "Remove high z redhsit for taylor analysis")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO? 1:CMB directon ")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole direction too?")
parser.add_option("--scanshelldip", action="store_true",default=False, dest="SCANSHELLDIPOLE")
parser.add_option("--fixvals", action="store_true",
                  default=True, dest="FIXPARS",help='Fixes parameters for Hd evalution (keep range 0.023-0.15)')
(options, args) = parser.parse_args()


if options.DET == 2:
    STYPE = 'NoScDep'
elif options.DET == 3:
    STYPE = 'Flat'
elif options.DET == 4:
    STYPE = 'Exp'
elif options.DET == 5:
    STYPE = 'Lin'
elif options.DET == 8:
    STYPE = 'mix'
else:
    STYPE = 'None'


if options.MET == 1:
    met = 'Nelder-Mead'
elif options.MET == 2:
    met = 'SLSQP'
elif options.MET == 3:
    met = 'Powell'
elif options.MET == 4:
    met = "trust-constr"
elif options.MET == 5:
    met = "TNC"
elif options.MET == 6:
    met = "COBYLA"
elif options.MET == 7:
    met = "L-BFGS-B"
elif options.MET == 8:
    met = "Newton-CG"
elif options.MET == 9:
    met = 'BFGS'
elif options.MET == 10:
    met = 'CG'
elif options.MET == 11:
    met = 'trust-exact'

c = 299792.458  # km/s
H0 = 70  # (km/s) / Mpc

CMBdipdec = -7
CMBdipra = 168

raLG_SUN = 333.53277784*u.deg
decLG_SUN = 49.33111122*u.deg
vLG_SUN = 299  # km/s


if options.FIXA > 0 and options.DET != 2:
    print('Error: need to give correction functional form for shell analysis')
    sys.exit(1)


def Minimizer(zlim1=0,dip=None):
    radip = None
    decdip = None
    name = str(met)
    Z = pd.read_csv('Zpan.csv')
    Z = Z.sort_values('zHEL')
    df = pd.read_csv('Pantheon+SH0ES.dat', delimiter=' ')

    if options.REVB:
        print('reversing bias')
        df = pd.read_csv('Pantheon+SH0ES.dat', delimiter=' ')
        Z['mB'] = Z['mB'] + df['biasCor_m_b']

    if options.TAY:
        print("Removing higher redshifts")
        Z=Z[Z['zHEL']<0.8]
        name+='TAYLOR_EXP_'

    if zlim1 != 0:
        print("Using the zlim provided", zlim1)
        Z = Z[Z['zHEL'] > zlim1]

    ra = np.array(df['RA'])*u.deg
    dec = np.array(df['DEC'])*u.deg

    # using redshift addition formula 
    zDF = np.sqrt((1-vLG_SUN*np.cos(ang(raLG_SUN, decLG_SUN, ra, dec)).value/c) /
                  (1+vLG_SUN*np.cos(ang(raLG_SUN, decLG_SUN, ra, dec)).value/c))-1


    zLG = ((1+df['zHEL'])/(1+zDF))-1

    l = Z.index.values.tolist()
    l = np.array(l)

    tempind = np.array([[3*i, 3*i+1, 3*i+2] for i in l])
    tempind = tempind.flatten(order='c')
    N = len(Z)
    Z['zLG'] = zLG
    Z = Z.to_numpy()

    if not options.DIP:
        name = name+"_FIX_DIP_TO_"
        if options.DIPDIR == 1:
            radip = CMBdipra
            decdip = CMBdipdec
            name = name+"CMB_"
    else:
        name = name+"_FLOAT_DIP"
    INDEX_DCT = {9: "zHEL", 0: "zHD", 10: "zCMB", 11: "ZLG"}

    ZINDEX=9
    if ZINDEX == 11:
        radip = 162.95389715
        decdip = -25.96734154

    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))

    def func(Zc, OM, OL, zh=None, zp=None):
        OK = 1.-OM-OL

        def I(z):
            return 1./np.sqrt(OM*(1+z)**3+OL+OK*(1+z)**2)
        if OK == 0:
            integ = integrate.quad(I, 0, Zc)[0]
        elif OK > 0:
            integ = (1./OK)**0.5 * \
                np.sinh(integrate.quad(I, 0, Zc)[0]*OK**(0.5))
        elif OK < 0:
            integ = (-1./OK)**0.5 * \
                np.sin(integrate.quad(I, 0, Zc)[0]*(-OK)**(0.5))
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

            k = 25 + 5*np.log10((c/H0)*dL_lcdm(Zc, OM, OL))

            np.save("MU_new", k)
        if np.any(np.isnan(k)):
            print('Fuck', OM, OL)
            k[np.isnan(k)] = 63.15861331456834

        return k

    def RES(OM, OL, A, B, M0, X0, C0):  # Total residual, \hat Z - Y_0*A
        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MU_lcdm(Z[:, ZINDEX], OM, OL)

        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    def MUZ(Zc, Q0, J0, S0=None, L0=None):
        k = 5.*np.log10(c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.
        if np.any(np.isnan(k)):

            k[np.isnan(k)] = 63.15861331456834
        return k

    # phenomenological taylor series expansion for dL from Visser et al

    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z - 1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.)*(1+Z[:, 9])/(1+z)


    def MUZ_H0Dip(Zc, Q0, J0, H0):
        k = 5.*np.log10(c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k

    # Total residual, \hat Z - Y_0
  
    def RESVF3_H0Dip(Hm,Hd, ra=radip, dec=decdip, A=0.1230, X0=-0.09398,B=2.076, C0=-0.01845, Q0=1.230e-01, J0=-1.123e+00,DS=np.inf, stype=STYPE):
        
        M0 = -19.3
        #Hd=0
        if options.FIXPARS:
            if ZINDEX == 10:
                Q0 = -7.937e-02
                J0 = -6.878e-01
            if ZINDEX == 0:
                Q0 = -1.623e-01
                J0 = -4.772e-01
            if ZINDEX == 11:
                Q0 = 1.410e-01
                J0 = -1.205e+00
        print(f'Q0:{Q0},J0:{J0},HM:{Hm},Hd:{Hd},ra:{ra},dec:{dec}')
        Zc = Z[:, ZINDEX]
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])

        if stype == 'NoScDep':
            H0 = Hm + Hd*cosangle
        elif stype == 'Flat':
            Hdip = Hd*cosangle
            H0[Zc > (DS+0.1)] = 0
            # minimizer steps are too small to probe an actual top hat
            H0[Zc > DS] = H0[Zc > DS]*np.exp(-1.*(Zc[Zc > DS]-DS)/0.03)
            H0 = Hm + Hdip
        elif stype == 'Exp':

            Hdip = Hd*cosangle*np.exp(-1.*Zc/DS)
            H0 = Hm + Hdip

        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MUZ_H0Dip(Z[:, ZINDEX], Q0, J0, H0)
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    # Total residual, \hat Z - Y_0
    def RESVF3_H0Dip_noscdep(Hd, M0, ra=radip, dec=decdip, Q0=1.230e-01, J0=-1.123e+00, A=0.1230, B=2.076, X0=-0.09398, C0=-0.01845, stype=STYPE):
        if ZINDEX == 10:
            Q0 = -7.937e-02
            J0 = -6.878e-01
        if ZINDEX == 0:
            Q0 = -1.623e-01
            J0 = -4.772e-01
        if ZINDEX == 11:
            Q0 = 1.410e-01
            J0 = -1.205e+00
        Hm=70
        print(f"Hd:{Hd},Hm:{Hm},RA:{ra},DEC:{dec}",
              "C0", C0, 'Q0:', Q0, 'J0:', J0)
        Zc = Z[:, ZINDEX]
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])

        if stype == 'NoScDep':
            H0 = Hm + Hd*cosangle
        else:
            print("WRONG STYPE")

        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MUZ_H0Dip(Z[:, ZINDEX], Q0, J0, H0)
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    # Total residual, \hat Z - Y_0
    def RESVF3_noscdep(qd, M0, ra=radip, dec=decdip, Q0=1.230e-01, J0=-1.123e+00, A=0.1230, B=2.076, X0=-0.09398, C0=-0.01845, stype=STYPE):
        if ZINDEX == 10:
            Q0 = -7.937e-02
            J0 = -6.878e-01
        if ZINDEX == 0:
            Q0 = -1.623e-01
            J0 = -4.772e-01
        if ZINDEX == 11:
            Q0 = 1.410e-01
            J0 = -1.205e+00
        Zc = Z[:, ZINDEX]
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])
        if stype == 'NoScDep':
            Q = Q0 + qd*cosangle
        else:
            print("WRONG STYPE")
        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MUZ(Z[:, ZINDEX], Q, J0)
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    name += str(INDEX_DCT[ZINDEX])+'_'
    print("Using Redshift:", INDEX_DCT[ZINDEX])
    zlim = Z[:, ZINDEX][0]

    COVd = np.load('statsyspan1.npy')[:, tempind][tempind]


    def COV(sM=0.075, A=1.20355320e-01, sX=9.57326949e-01, B=2.026, sC=7.73822769e-02, RV=0):  # Total covariance matrix

        # print("sM",sM,"sX",sX,"sC",sC,"A",A,"B",B)
        block3 = np.array([[sM**2 + (sX**2)*A**2 + (sC**2)*B**2,    -(sX**2)*A, (sC**2)*B],
                           [-(sX**2)*A, (sX**2), 0],
                           [(sC**2)*B,  0, (sC**2)]])
        ATCOVlA = linalg.block_diag(*[block3 for i in range(N)])

        if RV == 0:
            return np.array(COVd + ATCOVlA)
        elif RV == 1:
            return np.array(COVd)
        elif RV == 2:
            return np.array(ATCOVlA)

    def COVNosC(sM=1e-8, sX=0.9575, sC=0.07737, A=0.1230, B=2.076, RV=0):  # Total covariance matrix
        
        if ZINDEX == 11:
            sM = 0.075
        if ZINDEX == 9:
            sM = 0.067

        #sM=1e-8
        # print("sM",sM,"sX",sX,"sC",sC,"A",A,"B",B)
        block3 = np.array([[sM**2 + (sX**2)*A**2 + (sC**2)*B**2,    -(sX**2)*A, (sC**2)*B],
                           [-(sX**2)*A, (sX**2), 0],
                           [(sC**2)*B,  0, (sC**2)]])
        ATCOVlA = linalg.block_diag(*[block3 for i in range(N)])

        if RV == 0:
            return np.array(COVd + ATCOVlA)
        elif RV == 1:
            return np.array(COVd)
        elif RV == 2:
            return np.array(ATCOVlA)

    def RESVF3(M0,  A, X0, B, C0, Q0, J0, S0=None, L0=None):  # Total residual, \hat Z - Y_0*A

        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        mu = MUZ(Z[:, ZINDEX], Q0, J0, S0, L0)
        # print(mu)
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    # Total residual, \hat Z - Y_0*A
    def RESVF3Dip(M0,  A, X0, B, C0, Q0, J0, QD, DS=np.inf, ra=radip, dec=decdip, stype=STYPE):
        print("Q0:", Q0, "J0:", J0, "M0:", M0, "QD:", QD,
              "C0:", C0, "A:", A, "B:", B, "X0", X0, "DS:", DS)
        print(ra, dec)
        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])
        Zc = Z[:, ZINDEX]
        if stype == 'NoScDep':
            Q = Q0 + QD*cosangle
        elif stype == 'Flat':
            # print stype, QD, DS
            Qdip = QD*cosangle
            Qdip[Zc > (DS+0.1)] = 0
            # minimizer steps are too small to probe an actual top hat
            Qdip[Zc > DS] = Qdip[Zc > DS]*np.exp(-1.*(Zc[Zc > DS]-DS)/0.03)
            Q = Q0 + Qdip
        elif stype == 'Exp':
            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            # print 'Here', Qdip
            Q = Q0 + Qdip
        elif stype == 'Lin':
            Qd = QD - Zc*DS
            Qd[Qd < 0] = 0
            Q = Q0 + Qd*cosangle
        mu = MUZ(Zc, Q, J0)
        # print 'Now', Q0, J0, K, V0, V1
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    def RESVF3Quad(Hm, Q0, J0, QD, DS, DS1, lam1, lam2):
        print(f'Hm{Hm},QD:{QD},J0:{J0},lam1:{
              lam1},lam2:{lam2},DS:{DS},DS1:{DS1}')
        M0 = -19.30
        Zc = Z[:, ZINDEX]

        cosangle1 = cdAngle(194.18, 39.08, Z[:, 5], Z[:, 6])**2
        cosangle2 = cdAngle(248.52, -41.85, Z[:, 5], Z[:, 6])**2
        cosangle3 = cdAngle(306.43, 31.04, Z[:, 5], Z[:, 6])**2
        cosangle = cdAngle(167.94, -6.94, Z[:, 5], Z[:, 6])
        DS1 = 0.06/np.log(2)
        H0 = Hm*(1+(lam1*cosangle1+lam2*cosangle2 -
                 (lam1+lam2)*cosangle3)*np.exp(-Zc/DS1))
        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
        Q = Q0 + Qdip
        Y0A = np.array([M0])
        mu = MUZ_H0Dip(Z[:, ZINDEX], Q, J0, H0)
        return np.hstack([(Z[i, 1] - np.array([mu[i]]) - Y0A) for i in range(N)])

    # Total residual, \hat Z - Y_0*A
    def RESVF3Dip_temp(ra, dec, Q0, J0, A, B, M0, X0, C0, QD, DS=np.inf, stype=STYPE):
        print("Q0:", Q0, "J0:", J0, "M0:", M0, "QD:", QD,
              "C0:", C0, "A:", A, "B:", B, "X0", X0, "DS:", DS)
        print(ra, dec)
        Y0A = np.array([M0-A*X0+B*C0, X0, C0])
        cosangle = cdAngle(ra, dec, Z[:, 6], Z[:, 7])
        Zc = Z[:, ZINDEX]
        if stype == 'NoScDep':
            Q = Q0 + QD*cosangle
        elif stype == 'Flat':
            # print stype, QD, DS
            Qdip = QD*cosangle
            Qdip[Zc > (DS+0.1)] = 0
            # minimizer steps are too small to probe an actual top hat
            Qdip[Zc > DS] = Qdip[Zc > DS]*np.exp(-1.*(Zc[Zc > DS]-DS)/0.03)
            Q = Q0 + Qdip
        elif stype == 'Exp':
            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            # print 'Here', Qdip
            Q = Q0 + Qdip
        elif stype == 'Lin':
            Qd = QD - Zc*DS
            Qd[Qd < 0] = 0
            Q = Q0 + Qd*cosangle
        mu = MUZ(Zc, Q, J0)
        # print 'Now', Q0, J0, K, V0, V1
        return np.hstack([(Z[i, 1:4] - np.array([mu[i], 0, 0]) - Y0A) for i in range(N)])

    # Total residual, \hat Z - Y_0*A


    if options.EVAL == 1:
        function = lkly
    elif options.EVAL == 2:
        function = RESVF3
    elif options.EVAL == 3:
        print("Evaluating dipole")
        function = RESVF3Dip
    elif options.EVAL == 4:
        function = RESVF3_H0Dip
    elif options.EVAL == 5:
        function = RESVF3Q0_H0dip
    if options.FIXA > 0:
        op = options.FIXA
        if op == 1:
            function = RESVF3_noscdep
        else:
            function = RESVF3_H0Dip_noscdep


    def m2loglike(pars, RV=0):

        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')

        else:
            if options.FIXA == 0 and not options.FIXPARS:
                cov = COV(*[pars[i] for i in [1, 2, 4, 5, 7]])
            else:
                # cov = COVNosC(*[ pars[i] for i in [-3,-2,-1,6,7] ] )
                cov = COVNosC()

            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a=True, lower=True)
            except np.linalg.linalg.LinAlgError:  # If not positive definite
                print("LINALGERRROR error")
                return +13993*10.**20
            except ValueError:  # If contains infinity
                print("Value error")
                return 13995*10.**20

            if options.FIXA == 0 and not options.FIXPARS :

                res = function(
                    *[pars[i] for i, val in enumerate(pars) if i != 1 and i != 4 and i != 7])
            else:
                # res = function(*[pars[i] for i,val in enumerate(pars)  ])
                res = function(
                    *[pars[i] for i, val in enumerate(pars) if i != 10 and i != 11 and i != 12])
            # Dont throw away the logPI part.
            part_log = 3*N*np.log(2*np.pi) + \
                np.sum(np.log(np.diag(chol_fac[0]))) * 2
            part_exp = np.dot(res, linalg.cho_solve(chol_fac, res))

            if RV == 0:
                m2loglike = part_log + part_exp

                if options.SHOW:
                    print(pars, m2loglike)
                return m2loglike
            elif RV == 1:
                return part_exp
            elif RV == 2:
                return part_log

    def m2NODip(pars):
        return pars[10]

    pre_found_best = [-1.93109635e+01,  1.63904464e-08,  1.19973487e-01, -7.39643494e-02,
                      9.58814512e-01,  1.99686244e+00, -1.93103218e-02,  7.69038558e-02]
    bounds = ((None, None), (None, None), (None, None), (None, None),
              (None, None), (None, None), (None, None), (None, None))
    if options.EVAL == 1:
        name = name+" LCDM"
        ar = [0.3, 0.7]
        new = ((None, None), (None, None))
        bounds += ((None, None), (None, None))
        pre_found_best = np.hstack([pre_found_best, ar])
    else:
        ar = [-1.74546582e-01, -4.59347830e-01]
        bounds += ((None, None), (None, None))
        name = name+" TAY"
        pre_found_best = np.hstack([pre_found_best, ar])
        if options.PHENO4:
            ar = [-0.35]
            bounds += ((None, None),)
            name = name+"_pheno5"
            pre_found_best = np.hstack([pre_found_best, ar])
        if options.PHENO5:
            ar = [-0.35, 3.11]
            bounds += ((None, None), (None, None))
            name = name+"_pheno5"
            pre_found_best = np.hstack([pre_found_best, ar])
        if options.EVAL == 3:
            ar = [-14]
            bounds += ((None, None),)
            pre_found_best = np.hstack([pre_found_best, ar])
            if options.DET > 2:
                name = name+" F=f(z,s)"
                ar = [0.02]
                bounds += ((zlim, 1),)
                pre_found_best = np.hstack([pre_found_best, ar])
        if options.DIP:
            ar = [119, 34]
            name = name+" Estimated Dipole"
            bounds += ((0, 360), (-90, 90))
            pre_found_best = np.hstack([pre_found_best, ar])
        
        if options.EVAL == 4:
            pre_found_best = [71.02,  1.63904464e-08,  1.19973487e-01, -7.39643494e-02,
                              9.58814512e-01,  1.99686244e+00, -1.93103218e-02,  7.69038558e-02]
            bounds = ((65,75), (None, None), (None, None), (None, None),
                      (None, None), (None, None), (None, None), (None, None))
            
            if not options.FIXPARS:
                bounds = tuple(bounds)
                ar = [-3.40900046e-01]
                name = name+" H0dip"
                bounds += ((None, None),)
                pre_found_best = np.hstack([pre_found_best, ar])
                if options.DET > 2:
                    name = name+" F=f(z,S)"
                    ar = [0.02]
                    bounds += ((zlim, 1),)
                    pre_found_best = np.hstack([pre_found_best, ar])
            else:
                pre_found_best = [71.02,-1]
                bounds = ((65, 75),(None,None))
                #pre_found_best = [-19,-1]
                #bounds = ((-16, -21),(None,None))
            if options.DIP:
                   ar = [119, 34]
                   name = name+" Estimated Dipole"
                   bounds += ((0, 360), (-90, 90))
                   pre_found_best = np.hstack([pre_found_best, ar])
                
                
        

        # pre_found_best=np.array([-1.93095718e+01 , 5.06578998e-09,  1.20125959e-01, -7.39598561e-02,9.58836522e-01,  2.00711958e+00, -1.93111624e-02 , 7.68943705e-02,-1.92238507e-01, -3.86475938e-01, -3.40900046e-01] )
        # pre_found_best=np.hstack([pre_found_best,ar])
        if options.FIXA:
            emp = options.FIXA

            pre_found_best = [-1, -19.3]
            bounds = ((None, None), (None, None))
            if options.FIXA == 2:
                pre_found_best = [-1, 70]
                bounds = ((None, None), (None, None))
            if options.DIP:
                pre_found_best = [-2, 70, 130, 15, -0.55, 1,
                                  0.123, 2, -0.093, -0.018, 0.075, 0.957, 0.077]
                # pre_found_best=[-2,70,130,15]
                bounds = ((None, None), (None, None), (0, 360), (-90, 90), (None, None), (None, None), (None,
                          None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None))

        print(bounds)
        print(pre_found_best)

        def No_dip(pars):
            return pars[0]-dip
        if options.SCANSHELLDIPOLE:
            MLE = optimize.minimize(m2loglike, pre_found_best, method=met, tol=10**-14, options={
                                    'maxiter': 150000}, bounds=bounds, constraints=({'type': 'eq', 'fun': No_dip}))
        else:
            MLE = optimize.minimize(m2loglike, pre_found_best, method=met,
                                    tol=10**-14, options={'maxiter': 150000}, bounds=bounds)


        return MLE, name


if options.FIXA > 0:
    Z = pd.read_csv('Zpan.csv')
    Z = Z.sort_values('zHEL')
    Z = Z[Z['zHEL'] < 0.8]
    N = len(Z)
    shell_width = 100
    # m=Minimizer(zlim1=0.023)
    # print(m)
    
    if not options.SCANSHELLDIPOLE:
        num_splits = int(np.ceil(N/shell_width))
        init = 0
        ar = []
        ar2 = []
        mle = []
        for i in range(num_splits):
            final = shell_width*(i+1)
            l = Z.index.values
            m = Minimizer(init=init, final=final)
            print(m)
            init = final
            ar2.append(m[0].x[0])
            ar.append(m[0].x[1])
            mle.append(m[0].fun)

        print(ar)
        print(mle)
        print(f'x_true={ar2}\n ', f'MLE_true={mle} \n Hm={ar}')

    else:
        shell_width = 100
        ar = []
        fun = []
        num_splits = int(np.ceil(N/shell_width))
        init = 0
        # minimized_vals=[-38.86159740622688, -17.75354867779298, -8.208273573258992, -4.939017455520776, -4.178077420293118, -2.0032140695565652, -1.615233331256633, -0.7150906154827132, -0.1989928683734748, -0.044427387791486014, -0.14111061623341775, -0.3974002660681623, -0.0460520876681514, -0.11531701124053621, -0.2090319705703124, -0.011274688511929136, -0.057884518346162525]
        # MLE=[342.76973980211324, -57.65151182449563, -51.79681046385484, -87.71259784624718, -84.76411854459579, -97.83179766640306, -105.91089817848456, -107.31134428317338, -106.5501546363131, -85.36903374173403, -97.60385279482867, -102.87869020484587, -93.4912505041787, -109.57416129361042, -84.22649126879598, -63.06914507196426, -54.34585282946863]
        x_true=[-12.206207753914812, -7.012438540078619, -4.773289941547766, -3.360625228958532, -3.6637048645531824, -2.542420088090806, -2.842922015729894, -1.11902295794036, -0.6632403571296991, 0.06966307141820746, -0.7503889071820812, -2.782607021503099, -0.674097475268074, -1.487624894888768, -1.9879571613838085, -0.46972535894647593, -1.1052430386954093]
        MLE_true=[388.51025079176475, 91.64162074182514, 102.39154276859466, 49.91382780783749, 40.893422763189506, 44.60452938730782, -13.321629684808613, -36.402447660216495, -57.27331707234487, -46.153507093661005, -32.045702868895376, -53.18308994948936, -47.85951108597459, -44.471757973774544, -10.30501650069425, 8.269595710984078, -5.799629098143669]
        MLE = MLE_true
        minimized_vals = x_true
        if options.FIXA == 1:
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
                        minimized_vals[i]-0.5, minimized_vals[i]+0.5, 30))

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
                        minimized_vals[i]-3, minimized_vals[i]+3, 30))
                elif 6 <= i < 10:
                    dipole = list(np.linspace(
                        minimized_vals[i]-0.7, minimized_vals[i]+0.7, 30))
                else:
                    dipole = list(np.linspace(
                        minimized_vals[i]-0.3, minimized_vals[i]+0.3, 30))
    
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

    # Results already found


else:
    zlim=[0.00587, 0.00907, 0.01351, 0.01613 ,0.01826, 0.02121, 0.02324, 0.02531, 0.02873,
 0.03139, 0.03486, 0.04146, 0.05057, 0.06976, 0.10762] ## for tomographic cuts in steps of 50
    m = Minimizer(zlim1=0.00937)
    print(m)
    # name=m[-1]

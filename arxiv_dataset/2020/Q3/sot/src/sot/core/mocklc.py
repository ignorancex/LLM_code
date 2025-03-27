#!/usr/bin/python
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import healpy as hp
import time
import scipy.signal
from scipy.interpolate import splev, splrep
import tqdm

def comp_omega(nside):
    omega = []
    npix = hp.nside2npix(nside)
    #print(("npix=", npix))
    for ipix in range(0, npix):
        theta, phi = hp.pix2ang(nside, ipix)
        omega.append([theta, phi])
    return np.array(omega)


def uniteO(inc, Thetaeq):
    # (3)
    eO = np.array([np.sin(inc)*np.cos(Thetaeq), -
                   np.sin(inc)*np.sin(Thetaeq), np.cos(inc)])
    return eO


def uniteS(Thetaeq, Thetav):
    # (3,nsamp)
    eS = np.array(
        [np.cos(Thetav-Thetaeq), np.sin(Thetav-Thetaeq), np.zeros(len(Thetav))])
    return eS


def uniteR(zeta, Phiv, omega):
    # (3,nsamp,npix)
    np.array([Phiv]).T
    costheta = np.cos(omega[:, 0])
    sintheta = np.sin(omega[:, 0])
    cosphiPhi = np.cos(omega[:, 1]+np.array([Phiv]).T)
    sinphiPhi = np.sin(omega[:, 1]+np.array([Phiv]).T)
#    cosphiPhi=np.cos(omega[:,1]-np.array([Phiv]).T)
#    sinphiPhi=np.sin(omega[:,1]-np.array([Phiv]).T)

    x = cosphiPhi*sintheta
    y = np.cos(zeta)*sinphiPhi*sintheta+np.sin(zeta)*costheta
    z = -np.sin(zeta)*sinphiPhi*sintheta+np.cos(zeta)*costheta
    eR = np.array([x, y, z])

    return eR

def comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv):
    """
    Summary
    ------------
    Compute weight functions W_illuminated and W_visible
    
    """
    omega=comp_omega(nside)
    eO=uniteO(inc,Thetaeq)
    eS=uniteS(Thetaeq,Thetav)
    eR=uniteR(zeta,Phiv,omega)
    WV=np.einsum("ijk,i->jk",eR,eO)
    WV[WV<0.0]=0.0
    WI=np.einsum("ijk,ij->jk",eR,eS)
    WI[WI<0.0]=0.0
    return WI,WV

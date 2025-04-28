import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft,ifft,fftn,ifftn,fftfreq
from scipy.special import erf

import random
from functools import partial # for fixing some arguments in a function to return another function
import scipy.integrate as integ # for NFW velocity dispersion estimation
import pandas as pd
import time
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

import h5py
import re
import sys, os

# picasa
BASE_DIR = 'picasa/' # change this path if needed
sys.path.append(BASE_DIR + 'code/')
# setting up picasa
from picasa import PICASA
import gc
from cobaya.yaml import yaml_load_file
from cobaya.run import run

from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

pic = PICASA(base_dir=BASE_DIR)

###########################################################################################
class Basic(object):
    """Sets the basic quantities of interest related to the simulation, and defines some useful functions"""
    def __init__(self, lbox=300, pbc=False, mp=1, dim=3, nbox=30):
        """inputs:
        lbox: box size in Mpc/h (default is 300)
        pbc: periodic boundary conditions (default False)
        mp: particle mass in Msun/h (default is 1)
        dim: spatial dimension (default 3)
        nbox: number of sub-boxes along each side that the simulation is divided into (default is 30)"""
        self.lbox=lbox
        self.pbc=pbc
        self.mp=mp
        self.dim=int(dim)
        self.nbox=int(nbox)
        self.lsubbox=lbox/nbox
        
    def unfold(self,P):
        """ unfolds the spine, assuming periodic boundary conditions
        input: array of positions of tracers of the spine, shape [N,dim]
        output: array of same dimension, unfolded"""
        P_unfold=P.copy()
        for i in range(len(P)-1):
            fil1=P_unfold[i]
            fil2=P_unfold[i+1]
            fil2=np.subtract(fil2, fil1-self.lbox/2)%self.lbox-self.lbox/2+fil1
            P_unfold[i+1]=fil2
        return P_unfold
    
    def W(self,a,b,c):
        ''' gives the weight of the box given its x y and z indices and the number of cells along an axis'''
        a=a%self.nbox #periodic boundary conditions
        b=b%self.nbox
        c=c%self.nbox
        return (a*self.nbox**2+b*self.nbox+c)


    def rotate(self,pos, vec1, vec2):
        '''rotates the given position vectors pos so that vec1 has coordinates vec2 in the new coordinate system
        pos: vectors to be rotated with shape[3,N]
        vec1: coordinates of the vector in original coordinate system
        vec2: coordinates of the vector in new coordinate system'''

        dc=vec1/np.linalg.norm(vec1)
        B1=vec2/np.linalg.norm(vec2)

        V=np.cross(dc, B1)
        S=np.linalg.norm(V)
        C=np.dot(dc, B1)

        Vx=np.array([[0, -V[2], V[1]],[V[2], 0, -V[0]], [-V[1], V[0],0]], dtype=float)
        I=np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=float)

        if any(V):
            R=I+Vx+np.matmul(Vx, Vx)/(1+C) #rotation matrix
        else:
            R=I #when the two axes align

        Pos_rot=np.matmul(R, pos)
        return Pos_rot

################################################################################################
# curvature calculator
class ComputeCurvature(Basic):
    """ calculates the curvature of the discretely sampled spine """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def kappa(self, P):
        """ calculates the curvature of each segment of the spine
        input: array of positions of the tracers of the spine, shape [N, dim]
        output:
        Dictionary D with the following keys:
        kappa_f: kappa computed using outgoing direction (cannot be computed for the last segment)
        kappa_b: kappa calculated using incoming direction (cannot be computed for the first segment)
        kappa: total (average) kappa

        Note that the code unfolds the spine if periodic boundary conditions are applied before calculating the curvatures"""
        if (self.pbc is True):
            P=self.unfold(P)
        # calculating curvature
        kappa_f=np.zeros(len(P)-1)
        kappa_b=np.zeros(len(P)-1)
        for i in range(1,len(kappa_b)-1,1):
            v0=P[i]-P[i-1]
            v1=P[i+1]-P[i]
            v2=P[i+2]-P[i+1]
            l0=np.linalg.norm(v0)
            l1=np.linalg.norm(v1)
            l2=np.linalg.norm(v2)

            mu_f=np.dot(v1, v2)/(l1*l2)
            mu_b=np.dot(v0, v1)/(l0*l1)
            
            # if |mu| exceeds 1 due to numerical errors
            if (mu_f>1):
                mu_f=1
            elif (mu_f<-1):
                mu_f=-1
            theta_f=np.arccos(mu_f)

            if (mu_b>1):
                mu_b=1
            elif (mu_b<-1):
                mu_b=-1
            theta_b=np.arccos(mu_b)

            kappa_b[i]=theta_b/l1
            kappa_f[i]=theta_f/l1


        # for the first segment
        v1=P[1]-P[0]
        v2=P[2]-P[1]
        l1=np.linalg.norm(v1)
        l2=np.linalg.norm(v2)

        mu_f=np.dot(v1, v2)/(l1*l2)
        if (mu_f>1):
            mu_f=1
        elif (mu_f<-1):
            mu_f=-1
        theta_f=np.arccos(mu_f)
        kappa_f[0]=theta_f/l1

        # for the last segment
        v1=P[len(kappa_b)]-P[len(kappa_b)-1]
        v0=P[len(kappa_b)-1]-P[len(kappa_b)-2]
        l1=np.linalg.norm(v1)
        l0=np.linalg.norm(v0)

        mu_b=np.dot(v1, v0)/(l1*l0)
        if (mu_b>1):
            mu_b=1
        elif (mu_b<-1):
            mu_b=-1
        theta_b=np.arccos(mu_b)
        kappa_b[-1]=theta_b/l1


        # averaging
        kappa_tot=(kappa_f+kappa_b)/2
        kappa_tot[0]=kappa_f[0]
        kappa_tot[-1]=kappa_b[-1]

        D=dict()
        D["kappa_f"]=kappa_f
        D["kappa_b"]=kappa_b
        D["kappa"]=kappa_tot
        return D
    
    
################################################################################################################################
################PROFILE ESTIMATOR ################################################################################################
################################################################################################################################

# profile estimator
class ExtractProfiles(Basic):
    """ computes the phase space profiles """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.B=np.array([0,0,1], dtype=float) #Unit vector along the z-direction (filament axis)
    
    def create_rbins(self,rmin, rmax, dlogr=None, dr=None, N_r=10, log=True):
        """ creates linearly or logarithmically spaced radial bins.

        Inputs:
        rmin, rmax: minimum and maximum values of radial bins
        dlogr: logarithmic spacing
        dr: linear spacing
        Nr: number of bins [specify either dlogr (dr) or N_r in case of log (linear) spacing.] (default 10)
        If both are specified, dlogr (dr) will be chosen, and N_r will be ignored.
        If none are specified, default N_r will be used
        Note that if dlogr (dr) is specified, the upper edge of the last bin is not exactly rmax, is slightly larger, if the given bin width cannot split the range into integer number of bins.
        
        Outputs:
        Dictionary D with keys:
        "rbins": end points of the bins
        "rmid": mid point of the bins
        """

        if (log is True):
            if (dr is not None):
                print("\nlogarithmic binning is chosen, but linear spacing is specified!! ignoring the given value of dr")

            # creating the bins
            if (dlogr is not None):
                rbins=10**(np.arange(np.log10(rmin), np.log10(rmax)+dlogr, dlogr))
            else:
                rbins=np.logspace(np.log10(rmin), np.log10(rmax), N_r+1)

            # mid points of the bins
            rmid=np.zeros(len(rbins)-1)
            for i in range(len(rmid)):
                rmid[i]=np.sqrt(rbins[i]*rbins[i+1])


        else:
            if (dlogr is not None):
                print("\nlinear binning is chosen, but logarithmic spacing is specified!! ignoring the given value of dlogr ")

            # creating the bins
            if (dr is not None):
                rbins=np.arange(rmin, rmax+dr, dr)
            else:
                rbins=np.linspace(rmin,rmax, N_r+1)

            # mid points of the bins
            rmid=np.zeros(len(rbins)-1)
            for i in range(len(rmid)):
                rmid[i]=(rbins[i]+rbins[i+1])/2

        # creating the output dictionary
        D=dict()
        D["rbins"]=rbins
        D["rmid"]=rmid
        return D

    def create_zbins(self,lfil, N_z=20):
        """ creates linearly spaced bins along the axis of the filament.
        Inputs:
        lfil: length of the filament
        N_z: number of bins along the axis of the filament. (default is 20)

        Outputs:
        Dictionary D with keys:
        "zbins": end points of the bins
        "zmid": mid point of the bins"""

        zbins=np.linspace(0, lfil, N_z+1)
        zmid=np.zeros(N_z)
        for i in range(N_z):
            zmid[i]=(zbins[i]+zbins[i+1])/2
        D=dict()
        D["zbins"]=zbins
        D["zmid"]=zmid
        return D

    def straighten(self,Pos, P, rbins, Vel=None,sort=None, Num_cum=None, Num_W=None ):
        """straightens the curved filament by assuming that the filament spine is a piecewise continuous curve,
        created by straight line segments joining consecutive points in the discretely sampled spine.
        All the partilces in cylinders with rmin<r<rmax, and axes along these segments are rotated and translated so as to
        lie besides each other, forming a straight cylinderical filament.

        Inputs:
        Pos: positions of particles of the simulation box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P: discretely sampled points on the spine, in the shape [Ns, 3]
        rbins: radial bins
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles

        Outputs:
        Dictionary D with the following keys:
        Pos_straight: positions of particles in the straightened filament 
        Vel_straight: velocities of particles in the straightened filament
        lfil: length of the straightened filament
        """
        
        filx=P[:,0]
        Pos_straight=[]
        Vel_straight=[]
        z_start=0
        B=self.B

        for i in range(len(filx)-1):
            fil1=P[i] #first end point of the segment
            fil2=P[i+1] #second end point of the segment
            if(self.pbc is True):
                fil2=np.subtract(fil2, fil1-l/2)%l-l/2+fil1 #accounting for PBC, unfolding

            r_sep=fil2-fil1
            dc_length=np.linalg.norm(r_sep)
            if (dc_length==0):
                continue
            dc=r_sep/np.linalg.norm(r_sep) #direction cosines of the filament segment
            #############################################################################################################################
            #selecting sub-boxes to search for in case the simulation data is sorted
            if (sort is not None):
                #indices to search for
                minr=np.min([fil1, fil2], axis=0)
                maxr=np.max([fil1, fil2], axis=0)
                
                maxr=maxr+np.max(rbins)
                minr=minr-np.max(rbins)

                #floor is required to round down negative values
                indmax=np.array(np.floor(maxr/self.lsubbox), dtype=int) #maximum index of sub-boxes along each direction
                indmin=np.array(np.floor(minr/self.lsubbox), dtype=int)

                xind=np.array(np.arange(indmin[0], indmax[0]+1,1), dtype=int)
                yind=np.array(np.arange(indmin[1], indmax[1]+1,1), dtype=int)
                zind=np.array(np.arange(indmin[2], indmax[2]+1,1), dtype=int)

                X, Y, Z= np.meshgrid(xind, yind, zind)
                X=X.flatten()
                Y=Y.flatten()
                Z=Z.flatten()
                weights=self.W(X, Y, Z) #sub-boxes to search in this takes into account PBC

                X=Y=Z=0 #clear space

                weight0=weights[0] #first sub-box
                Pos2=Pos[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]
                if (Vel is not None):
                    Vel2=Vel[Num_cum[weight0]:Num_cum[weight0]+Num_W[weight0],:]

                for b in range(1,len(weights),1):
                    Pos1=Pos[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                    Pos2=np.concatenate([Pos2,Pos1],axis=0)

                    if(Vel is not None):
                        Vel1=Vel[Num_cum[weights[b]]:Num_cum[weights[b]]+Num_W[weights[b]],:]
                        Vel2=np.concatenate([Vel2,Vel1],axis=0)

            ######################################################################################################################
            else:  
                Pos2=Pos.copy()
                if (Vel is not None):
                    Vel2=Vel.copy()

            #translating to the frame of reference of the origin of the segment
            if(self.pbc is True):
                Pos2=(np.subtract(np.subtract(Pos2, (fil1-l/2))%l, l/2))
                fil2=(np.subtract(np.subtract(fil2, (fil1-l/2))%l, l/2))
                fil1=(np.subtract(np.subtract(fil1, (fil1-l/2))%l, l/2))
            else:
                Pos2=np.subtract(Pos2, fil1)
                fil2=np.subtract(fil2, fil1)
                fil1=np.subtract(fil1, fil1)


            #rotating so that the filament is along the z axis
            Pos_rot=self.rotate(np.transpose(Pos2), r_sep, B)
            r1_rot=self.rotate(fil1, r_sep, B)
            r2_rot=self.rotate(fil2, r_sep, B)
            Pos_rot=Pos_rot.T
            if (Vel is not None):
                Vel_rot=self.rotate(np.transpose(Vel2), r_sep, B)
                Vel_rot=Vel_rot.T

            #now, the segment is along the z axis
            minz=0 #minimum z of the segment
            maxz=r2_rot[2] #maximum z of the segment


            r_perp=np.sqrt((Pos_rot[:,0])**2+(Pos_rot[:,1])**2) #distance perpendicular to the spine
            ind1=np.where((Pos_rot[:,2]>=minz)&(Pos_rot[:,2]<maxz)&(r_perp>=rbins[0])&(r_perp<rbins[-1]))[0]
            Pos_rot=Pos_rot[ind1]
            Pos_rot[:,2]+=z_start
            Pos_straight.append(Pos_rot)
            z_start+=r2_rot[2] #the next filament will start at this z value.

            if (Vel is not None):
                Vel_rot=Vel_rot[ind1]
                Vel_straight.append(Vel_rot)


        # combining all the segments
        Pos_straight=np.concatenate(Pos_straight, axis=0)
        if (Vel is not None):
            Vel_straight=np.concatenate(Vel_straight, axis=0)

        else:
            Vel_straight=None

        D=dict()
        D["Pos_straight"]=Pos_straight
        D["Vel_straight"]=Vel_straight
        D["lfil"]=z_start
        return D


    def profile(self,Pos_straight, rmin, rmax, lfil, Vel_straight=None, dlogr=None, dr=None, N_r=10, N_z=20, N_phi=20,log=True, bin_in_phi=False):
        """ Calculates the 2d or 3d, as well as logitudinal and radial profiles for the given input.

        Inputs:
        Pos_straight: positions of particles in the box, aligned such that the spine of the filament is along the z-axis
                        shape is [Np, 3]
        rmin, rmax: bounds on radial bins
        lfil: length of the filament
        Vel_straight: velocities of the straightened filament, if present.
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins (default is 10)
        N_z: number of logitudian bins (default is 20)
        N_phi: number of azimuthat bins (default is 20)
        log: True for logarithmic bins in r, False for linear. Default is True.
        bin_in_phi: True if one needs 3D profiles (default is False)

        outputs:
        dictionary D with keys:
        "rho": 2D (3D) density profile, shape [N_r, N_z] ([N_r, N_z, N_phi])
        "rho_r": logitudinally averaged radial profile
        "rho_z": radially averaged longitudinal profile

        similarly for vz, vr, sigvz, sigvr in case Vel_straight is not None.
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""
        
        ZB=self.create_zbins(lfil, N_z)
        zbins=ZB["zbins"]
        zmid=ZB["zmid"]

        RB=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=RB["rbins"]
        rmid=RB["rmid"]

        if (bin_in_phi is True):
            phibins=np.linspace(0, 2*np.pi,N_phi+1)
            phimid=np.zeros(N_phi)
            for p in range(N_phi):
                phimid[p]=(phibins[p]+phibins[p+1])/2


        ##########################################################################################
        #defining arrays to store outputs
        # 2d (3d)  profiles
        if (bin_in_phi is True):
            rho=np.zeros([len(rmid), N_z, N_phi])
        else:
            rho=np.zeros([len(rmid), N_z])
        Num=rho.copy()
        Vol=rho.copy()
        vz=rho.copy()
        vr=rho.copy()
        sigvz=rho.copy()
        sigvr=rho.copy()

        # 1d profiles
        rho_r=np.zeros(len(rmid)) # radial density profile
        rho_z=np.zeros(N_z) # tangential density profile

        Vol_r=rho_r.copy()
        Vol_z=rho_z.copy()

        vz_r=rho_r.copy()
        vz_z=rho_z.copy()
        vr_r=rho_r.copy()
        vr_z=rho_z.copy()
        
        sigvz_r=rho_r.copy()
        sigvz_z=rho_z.copy()
        sigvr_r=rho_r.copy()
        sigvr_z=rho_z.copy()
        ################################################################################################
        # removing particles outside the cylinder of interest
        ind=np.where((Pos_straight[:,2]>=zbins[0])+(Pos_straight[:,2]<zbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        # r in cylindrical coordinates
        r_perp=np.sqrt(Pos_straight[:,0]**2+Pos_straight[:,1]**2)

        if (bin_in_phi is True):
            phi=np.arctan2(Pos_straight[:,1], Pos_straight[:,0])
            phi[phi<0]=phi[phi<0]+np.pi*2 # phi values from 0 to 2pi


        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=(Vel_straight[:,0]*Pos_straight[:,0]+Vel_straight[:,1]*Pos_straight[:,1])/r_perp #radial velocity

        ind=np.where((r_perp>=rbins[0])&(r_perp<rbins[-1]))[0]
        Pos_straight=Pos_straight[ind]
        r_perp=r_perp[ind]

        if (Vel_straight is not None):
            Vel_straight=Vel_straight[ind]
            Velr=Velr[ind]

        # calculating the profile
        for u in range(len(rmid)):
            ind=np.where((r_perp>=rbins[u])&(r_perp<rbins[u+1]))[0]
            Pos1=Pos_straight[ind]
            if (bin_in_phi is True):
                phi1=phi[ind]

            vol=lfil*np.pi*((rbins[u+1])**2-(rbins[u])**2)
            Vol_r[u]=vol
            rho_r[u]=len(Pos1[:,0])/vol

            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind]
                Velr1=Velr[ind]

                vz_r[u]=np.mean(Vel1[:,2])
                sigvz_r[u]=np.std(Vel1[:,2])
                vr_r[u]=np.mean(Velr1)
                sigvr_r[u]=np.std(Velr1)

            #looping over zbins
            for q in range(N_z):
                ind3=np.where((Pos1[:,2]>=zbins[q])&(Pos1[:,2]<zbins[q+1]))[0]
                Pos2=Pos1[ind3]

                if (bin_in_phi is False):
                    vol=np.pi*(rbins[u+1]**2-rbins[u]**2)*(zbins[q+1]-zbins[q])
                    rho[u][q]=len(Pos2[:,0])/vol
                    Num[u][q]=len(Pos2[:,0])
                    Vol[u][q]=vol

                    if ((Vel_straight is not None)&(len(Pos2[:,0])!=0)):
                        Vel2=Vel1[ind3]
                        Velr2=Velr1[ind3]

                        vz[u][q]=np.mean(Vel2[:,2])
                        sigvz[u][q]=np.std(Vel2[:,2])
                        vr[u][q]=np.mean(Velr2)
                        sigvr[u][q]=np.std(Velr2)

                else:
                    phi2=phi1[ind3]
                    if ((Vel_straight is not None)&(len(Pos2[:,0])!=0)):
                        Vel2=Vel1[ind3]
                        Velr2=Velr1[ind3]

                    for ph in range(N_phi):
                        ind4=np.where((phi2>=phibins[ph])&(phi2<phibins[ph+1]))[0]
                        Pos3=Pos2[ind4]

                        vol=np.pi*(rbins[u+1]**2-rbins[u]**2)*(zbins[q+1]-zbins[q])*(phibins[ph+1]-phibins[ph])/(2*np.pi)
                        rho[u][q][ph]=len(Pos3[:,0])/vol
                        Num[u][q][ph]=len(Pos3[:,0])
                        Vol[u][q][ph]=vol

                        if ((Vel_straight is not None)&(len(Pos3[:,0])!=0)):
                            Vel3=Vel2[ind4]
                            Velr3=Velr2[ind4]

                            vz[u][q][ph]=np.mean(Vel3[:,2])
                            sigvz[u][q][ph]=np.std(Vel3[:,2])
                            vr[u][q][ph]=np.mean(Velr3)
                            sigvr[u][q][ph]=np.std(Velr3)


        for t in range(N_z):
            ind2=np.where((Pos_straight[:,2]>=zbins[t])&(Pos_straight[:,2]<zbins[t+1]))[0]
            Pos1=Pos_straight[ind2]
            vol=(zbins[t+1]-zbins[t])*np.pi*((rbins[-1])**2-rbins[0]**2)
            Vol_z[t]=vol
            rho_z[t]=len(Pos1[:,0])/vol

            if ((Vel_straight is not None)&(len(Pos1[:,0])!=0)):
                Vel1=Vel_straight[ind2]
                Velr1=Velr[ind2]

                vz_z[t]=np.mean(Vel1[:,2])
                sigvz_z[t]=np.std(Vel1[:,2])
                vr_z[t]=np.mean(Velr1)
                sigvr_z[t]=np.std(Velr1)


        rho=rho*self.mp
        rho_z=rho_z*self.mp
        rho_r=rho_r*self.mp

        D=dict()
        D["rho"]=rho
        D["Num"]=Num
        D["Vol"]=Vol
        D["vz"]=vz
        D["vr"]=vr
        D["sigvr"]=sigvr
        D["sigvz"]=sigvz

        D["rho_r"]=rho_r
        D["rho_z"]=rho_z
        D["Vol_r"]=Vol_r
        D["Vol_z"]=Vol_z
        D["vz_r"]=vz_r
        D["vz_z"]=vz_z
        D["vr_r"]=vr_r
        D["vr_z"]=vr_z

        D["sigvz_r"]=sigvz_r
        D["sigvz_z"]=sigvz_z
        D["sigvr_r"]=sigvr_r
        D["sigvr_z"]=sigvr_z

        D["rbins"]=rbins
        D["rmid"]=rmid
        D["zbins"]=zbins
        D["zmid"]=zmid
        return D

    def estimate_profile(self,Pos, P, rmin, rmax, dlogr=None, dr=None, N_r=10, N_z=20, N_phi=20, log=True, Vel=None,
                         sort=None, Num_cum=None, Num_W=None, bin_in_phi=False):
        """ estimates the 2d and 1d profiles of filaments.
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P: discretely sampled points on the spine, in the shape [Ns, 3]
        rmin, rmax: inner and outer radius of the cylinder
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins (default 10)
        N_z: number of logitudian bins (default 20)
        N_phi: number of phi bins(default is 20)
        log: True for logarithmic bins in r, False for linear. Default is True.

        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles
        bin_in_phi:True if 3D profiles are required (default is False)

        Outputs:
        dictionary D with following keys:
        2d (3d) profiles: rho, vz, vr , sigvz, sigvr
        1d profiles: above quantities {q} with
        q_r: logitudinal averaged radial profiles
        q_z: radial averaged logitudinal profiles
        rbins, rmid, zbins, zmid: end points and mid points of radial and longitudinal bins respectively"""

        D=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=D["rbins"]

        D1=self.straighten(Pos, P, rbins, Vel,sort, Num_cum, Num_W)
        Pos_straight=D1["Pos_straight"]
        lfil=D1["lfil"]
        if (Vel is not None):
            Vel_straight=D1["Vel_straight"]
        else:
            Vel_straight=None

        D=self.profile(Pos_straight, rmin, rmax, lfil, Vel_straight, dlogr, dr, N_r, N_z, N_phi,log, bin_in_phi)
        return D

    def estimate_profile_curve(self,Pos, P, rmin, rmax, dlogr=None, dr=None,
                            N_r=None, N_z=None,N_phi=20, log=True,
                            nk=3,kappacuts=None, kappacut_type="absolute",
                            Vel=None,sort=None, Num_cum=None, Num_W=None, bin_in_phi=False):
        """ estimates the radial profiles, split by curvature into nk types.
        Inputs:
        Pos: positions of particles of the box, in the shape [Np, 3]  (sorted, in case sort is not None)
        P: discretely sampled points on the spine, in the shape [Ns, 3]
        rmin, rmax: inner and outer radius of the cylinder
        dlogr (dr): radial spacing for log (linear) radial bins
        N_r: number of radial bins
        N_z: number of logitudian bins
        N_phi: number of phi bins
        log: True for logarithmic bins in r, False for linear. Default is True.
        nk: number of parts the filaments is to be split into based on curvture.
        kappacuts: values of kappa that determine the boundaries of kappa bins
        This must include end points, so that len(kappa_cuts)=nk+1. 
        if None, segments divided into nk bins, based on percentiles in kappa values. (defalut is None)
        kappacut_type: "absolute" if kappacuts array gives absolute cuts in kappa, "percentile" if the array gives percentile values (then must be between 0 and 100)(default is "absolute")
        
        Vel: velocities of the particles in the shape [Np, 3] (sorted, in case sort is not None)
        sort, Num_cum, Num_W: information about the sorted particles
        bin_in_phi:True if 3D profiles are required (default is False)

        Outputs:
        list of nk+1 dictionaries with the following keys:
        Kmin, Kmax: cuts on curvature for the given bin
        2D (3D) profiles: rho, vr, sigvr, vz, sigvz
        radial profiles: rho_r, vr_r, sigvr_r (these are averaged over the segments which have the expected curvature of the bin)
        rbins, rmid, zbins, zmid: end points and mid points of radial bins
        
        The first dictionary is the total profile, not split by curvature."""

        D=self.create_rbins(rmin, rmax, dlogr, dr, N_r, log)
        rbins=D["rbins"]
        phibins=np.linspace(0, 2*np.pi,N_phi+1)

        # first, delete repetitive points
        dele=[]
        for i in range(len(P)-1):
            le=np.linalg.norm(P[i+1]-P[i])
            if (le==0):
                dele.append(i)
        P=np.delete(P, dele, axis=0)


        # calculating curvature
        K=ComputeCurvature(lbox=self.lbox, pbc=self.pbc, dim=self.dim)
        Dkappa=K.kappa(P)
        kappa_tot=Dkappa["kappa"]

        if (kappacuts is None):
            nperc=np.arange(0, nk+1, 1)
            percentiles=nperc/nk*100
            K=np.percentile(kappa_tot, percentiles)

        else:
            if (kappacut_type=="percentile"):
                K=np.percentile(kappa_tot, kappacuts)
            else:
                K=kappacuts
            nk=len(K)-1
            
        epsilon=1e-5
        K[0]=np.min(kappa_tot)
        K[-1]=np.max(kappa_tot)+epsilon # cuts on kappa


        K_BINS=[]
        for i in range(nk):
            ind=np.where((kappa_tot>=K[i])&(kappa_tot<K[i+1]))[0]
            K_BINS.append(ind)
       


        #now, straightening, but splitting
        filx=P[:,0]

        Pos_straight_split = [ [] for _ in range(nk) ]
        Vel_straight_split = [ [] for _ in range(nk) ]
        lfil_split=np.zeros(nk) # lengths of the split filaments
        LFIL=0 # total length of the filament, not split by curvature

        Pos_straight=[]
        Vel_straight=[]
        z_start=0
        B=self.B

        for i in range(len(filx)-1):
            Pnow=P[i:i+2] # this segment
            know=kappa_tot[i]
            Dnow=self.straighten(Pos, Pnow, rbins, Vel=Vel,sort=sort, Num_cum=Num_cum, Num_W=Num_W) # note that z values will overlap between segments, but we are only interested in radial profiles
            Pos_straight_now=Dnow["Pos_straight"]
            Vel_straight_now=Dnow["Vel_straight"]
            lnow=Dnow["lfil"]

            Pos_straight_now1=Pos_straight_now+np.array([0,0,LFIL])
            Pos_straight.append(Pos_straight_now1)
            Vel_straight.append(Vel_straight_now)
            LFIL+=lnow

            for w in range(nk):
                if (i in K_BINS[w]):
                    Pos_straight_now[:,2]+=lfil_split[w] # translating to the end of last segment
                    Pos_straight_split[w].append(Pos_straight_now)
                    Vel_straight_split[w].append(Vel_straight_now)
                    lfil_split[w]+=lnow


        #############################################################################################################
        OUT=[]
        Pos_straight=np.concatenate(Pos_straight, axis=0)
        if (Vel is None):
            Vel_straight=None
        else:
            Vel_straight=np.concatenate(Vel_straight, axis=0)

        D=self.profile(Pos_straight, rmin, rmax, LFIL, Vel_straight, dlogr, dr, N_r, N_z, N_phi,log, bin_in_phi=False) # not split by curvature
        D["lfil"]=LFIL
        OUT.append(D)
        for w in range(nk):
            ty=int(len(Pos_straight_split[w])) # number of segments with given curvature
            #print("bin in k",w,"number of segments:", ty)
            if (ty>0):
                Pos_straight=np.concatenate(Pos_straight_split[w], axis=0)

                if (Vel is not None):
                    Vel_straight=np.concatenate(Vel_straight_split[w], axis=0)
                else:
                    Vel_straight=None

                lfil=lfil_split[w]

                D=self.profile(Pos_straight, rmin, rmax, lfil, Vel_straight, dlogr, dr, N_r, N_z, N_phi,log, bin_in_phi=bin_in_phi)
                D["Kmin"]=K[w]
                D["Kmax"]=K[w+1]
                D["lfil"]=lfil
                OUT.append(D)

            else:
                if (bin_in_phi is False):
                    data2d=-1*np.ones([N_r, N_z])
                else:
                    data2d=-1*np.ones([len(rmid), N_z, N_phi])
                    
                datar=-1*np.ones(N_r)
                dataz=-1*np.ones(N_z)

                D={}
                D["rho"]=data2d
                D["Num"]=data2d
                D["Vol"]=data2d
                D["vz"]=data2d
                D["vr"]=data2d
                D["sigvr"]=data2d
                D["sigvz"]=data2d

                D["rho_r"]=datar
                D["rho_z"]=dataz
                D["Vol_r"]=datar
                D["Vol_z"]=dataz
                D["vz_r"]=datar
                D["vz_z"]=dataz
                D["vr_r"]=datar
                D["vr_z"]=dataz

                D["sigvz_r"]=datar
                D["sigvz_z"]=dataz
                D["sigvr_r"]=datar
                D["sigvr_z"]=dataz

                D["rbins"]=-1*np.ones(N_r+1)
                D["rmid"]=datar
                D["zbins"]=-1*np.ones(N_z+1)
                D["zmid"]=dataz

                D["Kmin"]=K[w]
                D["Kmax"]=K[w+1]
                D["lfil"]=0
                OUT.append(D)

        return OUT

################################################################################################################################
############################################## SPINE SMOOTHER  ###############################################################
############################################################################################################################

class SmoothSpine(ExtractProfiles):
    """ smooths the disperse spines, using different methods """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def FT(self,P, pad_type="odd", pad_len="full", n_interp="mindis"):
        """Fourier transforms the given curve, P, after linearly interpolating to get equispaced points
        inputs:
        P: positions of points on the spine [N,3]
        pad_type: type of padding, "odd" or "constant". (default is "odd")
        pad_len: length of padding ("full", "half" or "none") in terms of signal length. (defalut is "full")
        n_iterp: number of equispaced interpolating points on the spine, before fourier transforming. Either choose "mindis", "sameres" or an integer
        if "mindis", n_interp is such that distance between the points is equal to the length of the shortest segment.
        if "sameres", same number of points as input are used (default is "mindis")

        outputs:
        dictionary D with the following keys:
        Pf_pad: fourier transforms of the padded signal
        k_pad: wavenumber
        kmax, kmin: maximum and minimum wavenumbers
        start, end: starting and ending indices to cut at after inverse FT, to get original signal
        
        the function returns 0 if dimensions of P do not match input spatial dimensions dim"""
        #making sure that the dimensionality of the spine is correct
        if (len(P[0])!=self.dim):
            print("dimensions of spine does not match spatial dimensions!")
            return 0

        if (self.pbc is True):
            P=self.unfold(P)
        
        Pshift=np.roll(P, 1, axis=0)
        l=np.linalg.norm(P-Pshift, axis=1)
        l[0]=0
        d=np.cumsum(l) # distance along the spine

        if (n_interp=="sameres"):
            n_interp=len(P)
        elif (n_interp=="mindis"):
            lseg_min=np.min(l[1:]) # minimum length of the segment
            n_interp=int(np.max(d)/lseg_min)+1
        else:
            n_interp=int(n_interp)

        # interpolate
        d_inter=np.linspace(d[0], d[-1], n_interp) # evenly spaced distance
        d_s_inter=(d_inter[1]-d[0]) # mean interparticle spacing.
        
        F_pinter=interp1d(d, P, axis=0)
        P_inter=F_pinter(d_inter) # uniformly spaced points, linearly interpolated
        
        if (pad_len=="none"):
            k_pad=np.abs(fftfreq(len(d_inter), d_s_inter))
            Pfpad=fftn(P_inter, axes=0)
            Pl=[]
            Ph=[]
        else:
            # padding the signal
            if (pad_len=="full"):
                npad=len(d_inter)-1
            else:
                npad=int(len(d_inter)/2)

            dl_pad=np.flip(-d_inter[1:npad+1])
            dh_pad=d_inter[-1]+d_inter[1:npad+1]
            dpad=np.concatenate([dl_pad, d_inter, dh_pad])

            if (pad_type=="constant"):
                Pl=P_inter[0]*np.ones([len(dl_pad), self.dim])
                Ph=P_inter[-1]*np.ones([len(dh_pad), self.dim])

            else:
                Perr_l=P_inter[1:len(dl_pad)+1]-P_inter[0]
                Pl=np.flip(P_inter[0]-Perr_l, axis=0)
                Perr_h=P_inter[-len(dh_pad)-1:-1]-P_inter[-1]
                Ph=np.flip(P_inter[-1]-Perr_h, axis=0)
                
            Ppad=np.concatenate([Pl, P_inter, Ph], axis=0)
            #fourier transforming
            k_pad=np.abs(fftfreq(len(dpad), d_s_inter))
            Pfpad=fftn(Ppad, axes=0)

        D=dict()
        D["Pfpad"]=Pfpad
        D["k_pad"]=k_pad
        D["kmax"]=np.max(np.abs(k_pad))
        D["kmin"]=1/(len(d_inter)*d_s_inter) # corresponding to the non-padded signal
        D["start"]=len(Pl)
        D["end"]=len(Pl)+len(P_inter)
        return D
    
    def rth(self,k, k0, s_filt):
        """rounded top hat filter in fourier space, with cutoff wavenumber k0 and width of the apodizing Gaussian s_filt"""
        y=(erf((k0+k)/(np.sqrt(2)*s_filt))-erf((k-k0)/(np.sqrt(2)*s_filt)))/2
        return y
    
    def apply_rth(self,D, k0, s_filt):
        """applies the rth filter, inverse fourier transforms, cuts out the padding and gives the smooth signal.
        inputs:
        D: output of FT
        k0: cutoff wavenumber
        s_filt: width of apodizing Gaussian

        output:
        pnew: smooth spine in shape [N,3]"""
        
        filt_pad=self.rth(D["k_pad"], k0, s_filt)
        Pf_filt_pad=(D["Pfpad"].T*filt_pad).T
        P_filt_pad=ifftn(Pf_filt_pad, axes=0)
        pnew=np.real(P_filt_pad[D["start"]:D["end"]])
        return pnew
    

    def fou_smooth(self,P, k0, s_filt, pad_type="odd", pad_len="full",n_interp="mindis"):
        """Smooths the spine P in fourier space. 
        inputs:
        P: positions of points on the spine [N,3] (unfolded in case of PBC)
        k0: cutoff wavenumber
        s_filt: width of apodizing Gaussian
        pad_type: type of padding, "odd" or "constant". (default is "odd")
        pad_len: length of padding ("full", "half" or "none") in terms of signal length. (defalut is "full")
        n_interp: number of points on interpolated spine for FT
       
        output:
        pnew: smooth spine in shape [N,3]
        """
        D=self.FT(P, pad_type, pad_len, n_interp)
        pnew=self.apply_rth(D, k0, s_filt)
        return pnew
    
    def nei_smooth(self,P, Nsm):
        """Smooths the spine P using Neighbour smoothing.
        Inputs:
        P: input unsmoothed spine, of the shape [N,3] (unfolded in case of PBC)
        Nsm: number of smoothing iterations
        output:
        pnew: smooth spine in shape [N,3]"""
        
        #making sure that the dimensionality of the spine is correct
        if (len(P[0])!=self.dim):
            print("dimensions of spine does not match spatial dimensions!")
            return 0
        if (self.pbc is True):
            P=self.unfold(P)

        N=len(P)
        pnew=P.copy() # position after smoothing
        pold=P.copy()
        for q in range(Nsm):
            P1=np.roll(pold, -1, axis=0)
            P2=np.roll(pold, 1, axis=0)
            pnew=pold/2+P1/4+P2/4
            pnew[0]=P[0]
            pnew[-1]=P[-1]
            pold=pnew.copy()
        return pnew 
    
    ####################################################################################################
    # optimization
        
    # note that we have not added rho0 here, as it is completely degenerate with cf
    def gauss(self,r, s, cf):
        y=cf* np.exp(-r**2/(2*s**2))
        return y
    
    def opt_fou(self,P,Pos,rmin, rmax, op_file,
                pad_type="odd", pad_len="full",n_interp="mindis", s_filt_frac=5,
                N_z=20, N_r=20, N_phi=20, 
                kmax_frac=4,Nsamp=10, dsig=5,verbose=False, f_nseg=1,
               sort=None, Num_cum=None, Num_W=None):
        """ Smooths the spine P in fourier space optimally by minimizing the radius of the filament.
        Outputs the smoothed spine and optimum cutoff wavenumber

        inputs:
        P: position of tracers of the unsmoothed spine, shape[Nt,3] (unfolded in case of PBC)
        Pos: position of all the particles in the box, shape[N,3]
        rmin, rmax: minimum and maximum radii for the profile estimation
        op_file: output file to store the ASA data, must end with .txt
        pad_type: type of padding, ("odd" or "constant"),default "odd"
        pad_len:  length of padding ("full", "half" or "none") in terms of signal length. (defalut is "full")
        n_interp: number of interpolating points ("mindis" or "sameres")
        s_filt_frac: ratio between k0 and s_filt (default is 5)
        N_z, N_r, N_phi: bins for estimating profiles
        kmax_frac: if k0>kmax/kmax_frac, radius is set to a very large value. 
        This speeds up the minimization, as too high cutoff wavenumbers are discarded automatically. (default 4)
        Nsamp, dsig: parameters required for ASA, see picasa for details (default Nsamp=10, dsig=5)
        verbose: prints picasa outputs if True. (default is False)
        f_nseg: ratio of number of points on the smoothed to input spine. (default is 1)
        sort, Num_cum, Num_W: specified if the simulation box is divided into sub-boxes (this is required in optimal smoothing)
        output:
        Dictionary Dout with following keys:
        lfil_fou: length of fourier smoothed spine
        radius: minimum radius
        k0: cutoff wavenumber
        pnew: optimally smoothed spine"""
        large=1e3

        data=self.FT(P, pad_type=pad_type, pad_len=pad_len, n_interp=n_interp)
        data["Pos"]=Pos
        data["rmin"]=rmin
        data["rmax"]=rmax

        if (self.pbc is True):
            P=self.unfold(P)


        def cost_func(data,k0):
            Pos=data['Pos']
            rmin=data['rmin']
            rmax=data['rmax']
            kmax=data['kmax']/kmax_frac
            kmin=data['kmin']
            if ((k0>kmax)+(k0<kmin)):
                R=large
            else:
                mp=self.mp
                s_filt=k0/s_filt_frac
                pnew=self.apply_rth(data, k0, s_filt)
                Pshift=np.roll(pnew, 1, axis=0)
                l=np.linalg.norm(pnew-Pshift, axis=1)
                l[0]=0
                d=np.cumsum(l) # distance along the spine

                Nseg=int(f_nseg*len(P))
                d_in=np.linspace(d[0], d[-1], Nseg) # interpolating to required number of points
                F_pinter=interp1d(d, pnew, axis=0)
                pnew=F_pinter(d_in)
                # getting the radius
                D1=self.estimate_profile(Pos, pnew, rmin, rmax, N_r=N_r, N_z=N_z, N_phi=N_phi, log=True, Vel=None,
                         sort=sort, Num_cum=Num_cum, Num_W=Num_W, bin_in_phi=False)


                rho_r=D1["rho_r"]
                rho_z=D1["rho_z"]
                rmid=D1["rmid"]
                zmid=D1["zmid"]
                rbins=D1["rbins"]
                zbins=D1["zbins"]

                # getting poisson errorbars
                vol=np.zeros_like(rmid)
                dz=zbins[-1]-zbins[0]
                for i in range(len(vol)):
                    vol[i]=np.pi*(rbins[i+1]**2-rbins[i]**2)*dz

                num=np.array((rho_r/mp)*vol, dtype=int)
                rho_err=rho_r/np.sqrt(num)

                param, param_cov=curve_fit(self.gauss, rmid,rho_r, sigma=rho_err)

                R=np.abs(param[0])
            return R
        SEED=1
        PTYPE=1
        
        rng=np.random.RandomState(seed=SEED)
        dim=1 # number of parameters
        data_pack = {}
        
        ###########################
        # Needed for ASA 
        ###########################
        # ... mandatory

        data_pack['chi2_file'] = op_file
        data_pack['Nsamp'] = Nsamp
        data_pack['dsig'] = dsig
        data_pack['param_mins'] = [data['kmax']/5]
        data_pack['param_maxs'] = [data['kmin']]
        data_pack['model_func'] = cost_func
        data_pack['model_args'] = data

        ###########################
        # ... optional
        data_pack['last_frac'] = 0.25 if PTYPE==1 else 0.5
        data_pack['C_sig'] = 1.0 if PTYPE==1 else 2.0 # tails oversampled when C_sig > 1. needed for high-dim models.
        data_pack['eps_conv']=5
        data_pack['no_cost']=True # no need to provide 'data', 'invcov_data', 'cost_func' and 'cost_args' 
        data_pack['Ndata']=3 # usually length of the data, chi2 is assumed to have dof=Ndata-dim
        data_pack['rng'] = rng
        data_pack['verbose']=verbose
        TEST_STATS = False
        start_time = time.time()
        chi2, params, cov_asa, params_best, chi2_min, eigvals,rotate, flag_ok = pic.optimize(data_pack)
        pic.time_this(start_time)
        #####################################################################
        # comparing between quadratic fit optimum and sampled optimum
        k01=params_best
        s01=k01/5
        spine01=self.apply_rth(data, k01, s01)
        # getting the radius
        D1=self.estimate_profile(Pos, spine01, rmin, rmax, N_r=N_r, N_z=N_z, N_phi=N_phi, log=True, Vel=None,
                         sort=sort, Num_cum=Num_cum, Num_W=Num_W, bin_in_phi=False)
        rho_r=D1["rho_r"]
        rho_z=D1["rho_z"]
        rmid=D1["rmid"]
        zmid=D1["zmid"]
        rbins=D1["rbins"]
        zbins=D1["zbins"]

        # getting poisson errorbars
        vol=np.zeros_like(rmid)
        dz=zbins[-1]-zbins[0]
        for i in range(len(vol)):
            vol[i]=np.pi*(rbins[i+1]**2-rbins[i]**2)*dz

        num=np.array((rho_r/self.mp)*vol, dtype=int)
        rho_err=rho_r/np.sqrt(num)

        param, param_cov=curve_fit(self.gauss, rmid,rho_r, sigma=rho_err)
        R01=np.abs(param[0])
        
        # sampled minimum
        f=np.loadtxt(BASE_DIR+op_file)
        chi2_all=f[:,0]
        t_all=f[:,1]

        mi=np.where(chi2_all==np.min(chi2_all))[0]
        k02=t_all[mi[0]]
        R02=chi2_all[mi[0]]

        if (R01<=R02):
            k0_final=k01
            R0=R01
        else:
            k0_final=k02
            R0=R02
        #######################################################
        s_filt_final=k0_final/s_filt_frac
        pnew=self.apply_rth(data, k0_final, s_filt_final)
        # calculating the length of the spine
        pnew_shift=np.roll(pnew, 1, axis=0)
        l=np.linalg.norm(pnew-pnew_shift, axis=1)
        l[0]=0
        dthis=np.max(np.cumsum(l)) # distance along the spine
        
        Dout=dict()
        Dout["lfil_fou"]=dthis
        Dout["radius"]=R0
        Dout["k0"]=k0_final
        Dout["pnew"]=pnew
        
        return Dout
    
    
    
    def opt_nei(self, P, Pos,rmin, rmax, op_file, Nsm_max=1000,  
                 N_z=20, N_r=20, N_phi=20,
                 Nsamp=10, dsig=5,verbose=False):
        """ Smooths the spine P using neighbour smoothing optimally by minimizing the radius of the filament.
        Outputs the smoothed spine and optimum number of smoothings Nsm

        inputs:
        P: position of tracers of the unsmoothed spine, shape[Nt,3] (unfolded in case of PBC)
        Pos: position of all the particles in the box, shape[N,3]
        rmin, rmax: minimum and maximum radii for the profile estimation
        op_file: output file to store the ASA data, must end with .txt
        Nsm_max: maximum number of smoothings allowed The radius is set to a large value for Nsm>Nsm_max for faster convergence
        N_z, N_r, N_phi: bins for estimating profiles
        Nsamp, dsig: parameters required for ASA, see picasa for details (default Nsamp=10, dsig=5)
        verbose: prints picasa outputs if True. (default is False)
        output:
        Dictionary Dout with following keys:
        lfil_nei: length of fourier smoothed spine
        radius: minimum radius
        Nsm: optimum number of smoothings
        pnew: optimally smoothed spine"""

        if (self.pbc is True):
            P=self.unfold(P)

        
        large=1e3
        data=dict()
        data["Pos"]=Pos # adding new keys
        data["P_spine"]=P
        data["rmin"]=rmin
        data["rmax"]=rmax
        data["Nsm_min"]=0
        data["Nsm_max"]=Nsm_max
        def cost_func(data,Nsm):
            Nsm=int(Nsm)
            Pos=data['Pos']
            Pf=data["P_spine"]
            rmin=data['rmin']
            rmax=data['rmax']
            Nsm_min=data["Nsm_min"]
            Nsm_max=data["Nsm_max"]
            if ((Nsm>Nsm_max)+(Nsm<Nsm_min)):
                R=large
            else:
                mp=self.mp
                pnew=self.nei_smooth(Pf, Nsm)
                # getting the radius
                D1=self.estimate_profile(Pos, pnew, rmin, rmax, N_r=N_r, N_z=N_z, N_phi=N_phi, log=True, Vel=None,
                         sort=sort, Num_cum=Num_cum, Num_W=Num_W, bin_in_phi=False)
                
                rho_r=D1["rho_r"]
                rho_z=D1["rho_z"]
                rmid=D1["rmid"]
                zmid=D1["zmid"]
                rbins=D1["rbins"]
                zbins=D1["zbins"]

                # getting poisson errorbars
                vol=np.zeros_like(rmid)
                dz=zbins[-1]-zbins[0]
                for i in range(len(vol)):
                    vol[i]=np.pi*(rbins[i+1]**2-rbins[i]**2)*dz

                num=np.array((rho_r/self.mp)*vol, dtype=int)
                rho_err=rho_r/np.sqrt(num)

                param, param_cov=curve_fit(self.gauss, rmid,rho_r, sigma=rho_err)
                R=np.abs(param[0])
            return R

        SEED=1
        PTYPE=1
        rng=np.random.RandomState(seed=SEED)
        dim=1 # number of parameters
        data_pack = {}


        ###########################
        # Needed for ASA 
        ###########################
        # ... mandatory

        data_pack['chi2_file'] = op_file
        data_pack['Nsamp'] = Nsamp
        data_pack['dsig'] = dsig
        data_pack['param_mins'] = [0]
        data_pack['param_maxs'] = [Nsm_max]
        data_pack['model_func'] = cost_func
        data_pack['model_args'] = data

        ###########################
        # ... optional
        data_pack['last_frac'] = 0.25 if PTYPE==1 else 0.5
        data_pack['C_sig'] = 1.0 if PTYPE==1 else 2.0 # tails oversampled when C_sig > 1. needed for high-dim models.
        data_pack['eps_conv']=5
        data_pack['no_cost']=True # no need to provide 'data', 'invcov_data', 'cost_func' and 'cost_args' 
        data_pack['Ndata']=10 # usually length of the data, chi2 is assumed to have dof=Ndata-dim
        data_pack['rng'] = rng
        data_pack['verbose']=verbose
        #########################################################
        TEST_STATS = False
        start_time = time.time()
        chi2, params, cov_asa, params_best, chi2_min, eigvals,rotate, flag_ok = pic.optimize(data_pack)
        pic.time_this(start_time)
        #####################################################################
        # comparing quadratic optimum with sampled optimum
        Nsm1=int(params_best)
        spine01=self.nei_smooth(P, Nsm1)
        # getting the radius
        D1=self.estimate_profile(Pos, spine01, rmin, rmax, N_r=N_r, N_z=N_z, N_phi=N_phi, log=True, Vel=None,
                         sort=sort, Num_cum=Num_cum, Num_W=Num_W, bin_in_phi=False)

        rho_r=D1["rho_r"]
        rho_z=D1["rho_z"]
        rmid=D1["rmid"]
        zmid=D1["zmid"]
        rbins=D1["rbins"]
        zbins=D1["zbins"]

        # getting poisson errorbars
        vol=np.zeros_like(rmid)
        dz=zbins[-1]-zbins[0]
        for i in range(len(vol)):
            vol[i]=np.pi*(rbins[i+1]**2-rbins[i]**2)*dz

        num=np.array((rho_r/self.mp)*vol, dtype=int)
        rho_err=rho_r/np.sqrt(num)

        param, param_cov=curve_fit(self.gauss, rmid,rho_r, sigma=rho_err)

        R01=np.abs(param[0])

        #################################
        # sampled minimum
        f=np.loadtxt(BASE_DIR+op_file)
        chi2_all=f[:,0]
        t_all=f[:,1]

        mi=np.where(chi2_all==np.min(chi2_all))[0]
        Nsm2=int(t_all[mi[0]])
        R02=chi2_all[mi[0]]

        if (R01<=R02):
            Nsm0_final=Nsm1
            R0=R01
        else:
            Nsm0_final=Nsm2
            R0=R02
        #######################################################

        pnew=self.nei_smooth(P, Nsm0_final)
        
        pnew_shift=np.roll(pnew, 1, axis=0)
        l=np.linalg.norm(pnew-pnew_shift, axis=1)
        l[0]=0
        dthis=np.max(np.cumsum(l)) # distance along the spine
        
        Dout=dict()
        Dout["lfil_nei"]=dthis
        Dout["radius"]=R0
        Dout["Nsm"]=Nsm0_final
        Dout["pnew"]=pnew
        
        return Dout

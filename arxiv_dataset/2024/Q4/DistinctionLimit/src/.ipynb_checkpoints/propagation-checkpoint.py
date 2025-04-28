#===========================propagation.py===================================#
# Created by Bing-Long Zhang, 2024

# Contains functions for performing calculations of the propagation of neutrinos from Sun to Earth

#==============================================================================#
# import
import sys, os
# current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# relative_path = os.path.join(parent_dir, 'src')
# sys.path.append(relative_path)
import numpy as np
from scipy import interpolate
#==============================================================================#
# Neutrino Data
neutrinoFluxDir = os.path.join(parent_dir, 'data', 'neutrinos', 'normalised')
neutrinoDisDir = os.path.join(parent_dir,'data','solar')
neutrinoNames = np.array(["pp", "pep", "hep", "7Be1", "7Be2", "8B", "13N", "15O", "17F"])
neutrinoFluxesNorFactors = np.array([5.98e10,1.44e8,7.98e3,4.93e8,4.50e9,5.16e6,2.78e8,2.05e8,5.29e6]) # MeV
neutrinoFluxesUnc = np.array([0.006, 0.01, 0.3,0.06, 0.06, 0.02, 0.15 ,0.17 ,0.2])
neutrinoNorFluxes = list(map(lambda name: np.loadtxt(os.path.join(neutrinoFluxDir,name+'.txt'),delimiter=','), neutrinoNames))
neutrinoFluxes = list(map(lambda flux: np.transpose([flux[0][:,0], flux[0][:,1]*flux[1] ]), zip(neutrinoNorFluxes, neutrinoFluxesNorFactors)))

electronDistribution = np.loadtxt(os.path.join(neutrinoDisDir,'SSM_BS_2005_ELECTRON.txt'))
HHeDistributionsDat = np.loadtxt(os.path.join(neutrinoDisDir,'SSM_BS_2005.txt'))
HHeDistributions = list(map(lambda i: np.transpose([HHeDistributionsDat[:,0], HHeDistributionsDat[:,i]]), [6,7]))
neutrinoDistributionsDat = np.loadtxt(os.path.join(neutrinoDisDir,'SSM_BS_2005_NEUTRINO_FLUX.txt'))
neutrinoDistributions = list(map(lambda i: np.transpose([neutrinoDistributionsDat[:,0], neutrinoDistributionsDat[:,i]]), [5,11,12,10,10,6,7,8,9]))
def norFunc(dat):
    x,y = np.transpose(dat)
    #print(len(x[1:]),len(x[:]))
    norFactor = np.sum((x[1:]-x[:-1])*(y[:-1]+y[1:])/2)
    return np.transpose([x,y*1/norFactor])
neutrinoDistributions = list(map(norFunc, neutrinoDistributions))

neFunc = np.vectorize(interpolate.interp1d(electronDistribution[:,0], 10**electronDistribution[:,1]))
YnList = HHeDistributions[1][:,1]/(2*HHeDistributions[0][:,1]+HHeDistributions[1][:,1])
YuFunc = np.vectorize(interpolate.interp1d(HHeDistributions[0][:,0], 2+YnList))
YdFunc = np.vectorize(interpolate.interp1d(HHeDistributions[0][:,0], 1+2*YnList))
neutrinoDistributionFuncs = list(map(lambda x: np.vectorize(interpolate.interp1d(x[:,0], x[:,1])), neutrinoDistributions))

neutrinoFluxes = list(map(lambda x: np.transpose([0.001*x[:,0], 1000*x[:,1]]) if x[1,0]>0 else np.transpose([0.001*x[:,0], x[:,1]]), neutrinoFluxes))
neutrinoFluxesInt = list(map(lambda x: np.concatenate(([[1.0e-15,0],[0.9999*x[0,0],1.0e-10]],x,\
                                                    [[1.0001*x[-1,0],1.0e-10],[1.0e5,0]]),axis=0) if x[1,0]>0 else x, neutrinoFluxes))
neutrinoFluxFuncs = list(map(lambda x: np.vectorize(interpolate.interp1d(x[:,0], x[:,1])) if x[1,0]>0 \
                             else np.vectorize(lambda y: x[0,1] if np.abs(y-x[0,0])<1.e-12 else 0.), neutrinoFluxesInt))
#==============================================================================#
# Propagation Functions
#==============================================================================#
rList = np.linspace(0.00165,0.499,1000)
neList = neFunc(rList)
YuList = YuFunc(rList)
YdList = YdFunc(rList)
YqList = np.array([YuList, YdList, np.ones(len(rList))])
rhoList = [func(rList) for func in neutrinoDistributionFuncs]
theta12 = np.arcsin(np.sqrt(0.31))
theta23 = np.arcsin(np.sqrt(0.582))
theta13 = np.arcsin(np.sqrt(0.0224))
Dm212 = 7.39e-5
Dm312 = 2.525e-3
[s12, s23, s13, s212] = np.sin([theta12, theta23, theta13, 2*theta12])
[c12, c23, c13, c212] = np.cos([theta12, theta23, theta13, 2*theta12])

GF = (7.63e-14)/np.sqrt(2)

def rotate(direction, theta):
    c, s = np.cos(theta), np.sin(theta)
    return [np.array([[1.,0.,0.],[0.,c,s],[0.,-s,c]]), np.array([[c,0.,s],[0.,1.,0.],[-s,0,c]]), np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])][direction]

U1mat = np.dot(rotate(0, theta23), rotate(1, theta13))
U1matT = np.transpose(U1mat)
#==============================================================================#
def U1toNSI(modelPara):
    G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
    mat = np.zeros((3,3,3))
    for i in range(len(modelPara[2])):
        for j in range(len(modelPara[2])):
            mat[i,j] = modelPara[2][:,i,j]
    return modelPara[0]**2/(np.sqrt(2)*G_F_GeV*modelPara[1]**2)*mat

# The PeeGen has been modified: i) epDList; ii) PeeList.
def Pee(EvList, epQ):
    epDList = -(c13**2)/2*epQ[0,0] + (c13**2-(s23**2-s13**2*c23**2))/2*epQ[1,1] + (s23**2-c23**2*s13**2)/2*epQ[2,2] + \
        s13*c13*s23*epQ[0,1] + s13*c13*c23*epQ[0,2] - (1+s13**2)*c23*s23*epQ[1,2]
    epNList = -s13*c23*s23*epQ[1,1] + s13*c23*s23*epQ[2,2] + c13*c23*epQ[0,1] - c13*s23*epQ[0,2] +s13*(s23**2-c23**2)*epQ[1,2]
    
    def subFunc(Ev):
        AList = 4*np.sqrt(2)*Ev*GF*neList*((c13**2)/2-np.sum([YqList[i]*epDList[i] for i in range(len(epDList))], axis=0))
        BList = 4*np.sqrt(2)*Ev*GF*np.sum([neList*YqList[i]*epNList[i] for i in range(len(epDList))], axis=0)
        cos2thetaMList = (Dm212*c212-AList)/np.sqrt((Dm212*c212-AList)**2+(Dm212*s212+BList)**2)
        #print(AList,' ',BList,' ',cos2thetaMList)
        PeeList = (c13**4)*((1+cos2thetaMList*c212)/2)+s13**4
        return PeeList
    return np.array(list(map(subFunc,  EvList*1.0e9)))

def U1toNSIKineticMxing(gzp, mzp, chargeList):
    G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
    alphaEM = 1/137.036
    mE, mMuon, mTau = 0.511*1e-3, 105.66*1e-3, 1776.86*1e-3
    quA, qdA, qLA, qLp = [2/3, -1/3 ,-1 ,np.array(chargeList)]
    epuv, epdv, epLv = np.diag(quA*qLp), np.diag(qdA*qLp), np.diag(qLA*qLp)
    modelMat = np.array([U1matT@mat@U1mat for mat in [epuv, epdv, epLv]])
    mat = np.zeros((3,3,3))
    for i in range(len(modelMat)):
        for j in range(len(modelMat)):
            mat[i,j] = modelMat[:,i,j]
    
    ep = ((-np.array(chargeList))@np.log([mE, mMuon, mTau]))/3
    #ep = np.log(mTau/mMuon)/3
    #epDict = dict(zip([(1,-1,0),[1,0,-1],[0,1,-1]], oneLoopCouplingFunc))
    #ep = epDict(chargeList)
    U1toNSIFactor = gzp**2*np.sqrt(2)*alphaEM/(np.pi*G_F_GeV*(mzp**2))*ep
    return U1toNSIFactor*mat

class propagation(object):
    
    def __init__(self, modelPara, EvList=np.logspace(-2,np.log10(2*10.),200)*1.0e-3):
        gzp, mzp, chargeList = modelPara
        self.EvList = EvList
        proMat = U1toNSIKineticMxing(gzp, mzp, chargeList)
        self.PeeList = Pee(EvList, proMat)
    
    def PeeEvTest(self, name, Ev):
        rhoListT = rhoList[np.where(neutrinoNames==name)[0][0]]
        PeeListList = self.PeeList
        Pee = np.array([np.trapz(PeeList*rhoListT, rList) for PeeList in PeeListList])
        return np.interp(Ev,self.EvList,Pee)
    
    def PeeEv(self, name, Ev, flux):
        rhoListT = rhoList[np.where(neutrinoNames==name)[0][0]]
        PeeListList = self.PeeList
        PeeTemp = np.array([np.trapz(PeeList*rhoListT, rList) for PeeList in PeeListList])
            
        if flux[1]>0.:
            Pee = np.interp(Ev,self.EvList,PeeTemp)
            Pemu = (1-Pee)*c23**2
            Petau = (1-Pee)*s23**2
            res = np.array([P*flux for P in [Pee, Pemu, Petau]])
        else:
            Pee = np.interp(Ev[0],self.EvList,PeeTemp)
            Pemu = (1-Pee)*c23**2
            Petau = (1-Pee)*s23**2
            res = np.array([P*flux for P in [Pee, Pemu, Petau]])
        #print([flux[:3],res[:,:3]])
        return res


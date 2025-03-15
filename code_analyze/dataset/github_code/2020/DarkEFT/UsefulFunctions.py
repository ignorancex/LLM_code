#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################## ##############################
#########################  Various useful function  #########################
######################### ######################### #############################

### This modules defines various useful functions for modifying, summing, loading or manipulating an array of limits

### L. Darme, S. Ellis, T. You, 29/12/2019



############################ Messy library import part #########################
import numpy as np
from scipy import optimize as opt


################################################## #########################
#########################  Various Useful function, plot/sample       #########################
######################### ######################### #########################

s=-1

#---------- Cleaning an array

def clean_array(yi, smallval=1e-5):

    
    infpoint=np.isinf(yi)
    zero=np.zeros(np.size(yi))+smallval
    yi[infpoint]=zero[infpoint]
    
    Yif = np.nan_to_num(yi)+smallval*smallval

    return(Yif)


#---------- Log sampling before two values 
def log_sample(min, max, point=50):
    xi = np.logspace(np.log(min)/np.log(10), np.log(max)/np.log(10), num=point)
    return xi

#---------- Combine a lower and an upper limit curves, MinSplit define when we merge the two curves
def CombineUpDown(xi,yd,yup,MinSplit=1.1,thresholdlow=1,thresholdhigh=1):
     # Threshold high is used only if the calculation of the lower limit crashed

    if np.count_nonzero((np.nan_to_num(yup,posinf=0)>1))==0: 
        return (xi,yup)
    if np.count_nonzero((np.nan_to_num(yd,posinf=0)>thresholdlow))==0: 
        return (xi,yup)
#     print(np.nan_to_num(yup))
    indexs = xi.argsort()
    Yup = clean_array(yup)
    Yd =  clean_array(yd)
    
    gpth=np.logical_and(np.logical_and(Yup>thresholdlow,np.logical_or(Yup>thresholdhigh,Yd>thresholdlow)) , Yd*MinSplit<Yup )
    yuptmp= Yup[gpth]
    xuptmp= xi[gpth]
    gpth=np.logical_and(Yd>thresholdlow, Yd*MinSplit<Yup )
    ydres= Yd[gpth]
    xd=xi[gpth]

    # Cutting the point of up when down stops
    yupres=yuptmp[xuptmp<np.max(xd)]
    xup=xuptmp[xuptmp<np.max(xd)]  
    xres=np.concatenate((xup,np.flip(xd,axis=0)),axis=0)
    yres=np.concatenate((yupres,np.flip(ydres,axis=0)),axis=0)
                      
    return xres,yres



#---------- Combine two limits with upper and lower, keep the best
def CombineTwoLimitsUpDown(x1i,y1din,y1upin,x2i,y2din,y2upin):
     # Threshold high is used only if the calculation of the lower limit crashed

    indexs = x1i.argsort()  
    min1=np.min(x1i);min2=np.min(x2i)
    max1=np.max(x1i);max2=np.max(x2i)
    minall=np.min(min1,min2)
    maxall=np.max(max1,max2)
    xfull=log_sample(minall,maxall,200)
    
    y1up=np.interp(xfull,x1i,y1upin)
    y1d=np.interp(xfull,x1i,y1din)
    y2up=np.interp(xfull,x2i,y2upin)
    y2d=np.interp(xfull,x2i,y2din)
#  We generate the full support in x and prepare the data NFull
    yup = np.max(y1up,y2up)
    ydown=np.min(y1d,y2d)
#   
    
    xres,yres=CombineUpDown(xfull,ydown,yup)       
                        
    return xres,yres

#---------- Combine limits with potentially different support, keep the best (ie highest)
def CombineLimits(xall,yall):
  
    #   We find the min and max of the whole distribution, taking care of the one-dimensinal case
    minall=xall[1][1]
    maxall=xall[1][1]
    for xdat in xall:
        if (np.min(xdat)<minall):
            minall=np.min(xdat)
        if (np.max(xdat)>maxall):
            maxall=np.max(xdat)
    #  We generate the full support in x 
    xi = log_sample(minall,maxall,200)
    LimFull=np.zeros(200)
# Run over each limit, interpolate it on its support
    for (xdat,ydat) in zip(xall,yall):
        ydat[ydat == 0] = 1e-40
        ydat[ydat < 1.e-40] = 1e-40

        Limtmp=np.interp(xi, xdat, ydat)
#         print(LimFull,Limtmp)
        LimFull=np.maximum(np.nan_to_num(LimFull),np.nan_to_num(Limtmp))
  
    return xi,LimFull

################################################## #########################
#########################  Loading data for this project      #########################
######################### ######################### #########################


#---------- Standard data loading for this project, automatically interpolates to 200 points
def LoadData(file,type="log"):

    data =  np.loadtxt('Data/'+file)
    xi = data[:,1+s]
    CSData = data[:,2+s]
    if type =="log":
        mXData=np.abs(xi)
        xi_Direct = log_sample(np.min(mXData),np.max(mXData),200)
        CS_FDM_interp = np.interp(xi_Direct, mXData, CSData)
    if type =="lin":
        xi_Direct = np.linspace(np.min(xi),np.max(xi),num=200)
        CS_FDM_interp = np.interp(xi_Direct, xi, CSData)
        
    return(xi_Direct,CS_FDM_interp)

def LoadDirectProd(file,mygs,pnud,pndu,A,Z,PoT,isHLLHC=False):
    
    gu=mygs.get("gu11",0.);gd=mygs.get("gd11",0.)
    
    xuu,CSuu= LoadData("ProdData/"+file+"_uu.txt")
    xdd,CSdd= LoadData("ProdData/"+file+"_dd.txt")
    
    if isHLLHC:
        NP=PoT*(gu*gu*CSuu+gd*gd*CSdd)
    else:
        CSpn=gu*gu*pnud*CSuu+gd*gd*pndu*CSdd # pnud is the fraction between the proton proton CS for uu partons and the proton neutron one
        CSpp=gu*gu*CSuu+gd*gd*CSdd
        CSpADark=Z/A*CSpp + (A-Z)*CSpn/A # Dark cross section
        CSpA = (40e9) # Proton full inelastic cross-section for this experiment 
        NdirectperPoT=CSpADark/CSpA
        NP=PoT*NdirectperPoT
   
    return xuu,NP
    
    
def LoadDirectProdDP(file,A,Z,PoT,isHLLHC=False):
      
    xuu,CSem= LoadData("ProdData/"+file+".txt")     
    if isHLLHC:
        NP=PoT*CSem
    else:
        CSpA = (40e9) # Proton full inelastic cross-section for this experiment 
        NdirectperPoT=(A+Z)/2/A*CSem/CSpA
        NP=PoT*NdirectperPoT    
    return xuu,NP
  

#---------- Specialised loading function to get the data from BdNMC mesons + Brem dark photon production and sum them into a curve
def LoadandSumData(file):
    data =  np.loadtxt('Data/'+file)
    mXData = np.abs(data[:,1+s])
    N1Data = data[:,2+s]
    N2Data = data[:,3+s]
    N3Data = data[:,4+s]
    N4Data = data[:,5+s]
    N5Data = data[:,6+s]

    xi_Direct = log_sample(np.min(mXData),np.max(mXData),200)
    CS_FDM_interp = np.interp(xi_Direct, mXData, N1Data+N2Data+N3Data+N4Data+N5Data)

    return(CS_FDM_interp,xi_Direct)

# ----- Get the ratio between two quantities while properly interpolating them, used to get the initial limits from Dark Photon to the effective theory case
def GetRatio(CS1,x1,CS2,x2,lim,xlim):
    # Making sure we take the ratio and limit only where its properly defined
    newmin=np.max((np.min(x1),np.min(x2),np.min(xlim)))
    newmax=np.min((np.max(x1),np.max(x2),np.max(xlim)))

    xi_Direct = log_sample(newmin,newmax,200)
    CS1int = np.interp(xi_Direct, x1, CS1)
    CS2int = np.interp(xi_Direct, x2, CS2)
    Limint = np.interp(xi_Direct, xlim, lim)
    
    gp=(CS2int>0)
    ratio=np.zeros(np.shape(CS1int))
    ratio[gp]= CS1int[gp]/CS2int[gp]

    return(xi_Direct,ratio,Limint)

### ---- As the name suggest, this sums three datasets, and put the rest to zero, only used once, should be replaced by the more generic function below
def SumInterpZero(xdat,ydat,xdat2,ydat2,xdat3,ydat3):

    zero=np.zeros(50)

    x=log_sample(np.min(xdat3),np.max(xdat3),2000)

    CS1 = np.interp(x, xdat, ydat)
    CS2 = np.interp(x, xdat2, ydat2)
    CS3 = np.interp(x, xdat3, ydat3)
    CS1[CS1 < np.min(CS1)*1.01] = 0
    CS2[CS2 < np.min(CS2)*1.01] = 0
    CS3[CS3 < np.min(CS3)*0.85] = 0

    CSFull=CS1+CS2+CS3
    return(x,CSFull)

### --- Sum an arbitrary number of channels, with different supports in terms of masses
def SumInterp2(xdatA,ydatA,fillvalue=1e-40):
    
#   We find the min and max of the whole distribution, taking care of the one-dimensinal case
    if np.ndim(xdatA[1])==0:
        minall=xdatA[1]
        maxall=xdatA[1]
    else:
        minall=xdatA[1][1]
        maxall=xdatA[1][1]
        for xdat in xdatA:
            if (np.min(xdat)<minall):
                minall=np.min(xdat)
            if (np.max(xdat)>maxall):
                maxall=np.max(xdat)
#  We generate the full support in x and prepare the data NFull
    xlog = np.linspace(np.log(minall)/np.log(10.), np.log(maxall)/np.log(maxall),200)
    NFull=np.zeros(200)
    
# Run over each production channel, interpolate it on its support and add it to the full production array    
    if np.ndim(xdatA[1])>0:

        for (xdat,ydat) in zip(xdatA,ydatA):
            ydat[ydat == 0] = 1e-40
            ydat[ydat < 1.e-40] = 1e-40

            Ntmp=np.zeros(200)
            logydat=np.log(ydat.astype('float64'))/np.log(10.)
            logxdat=np.log(xdat)/np.log(10.)
            xlogin=xlog[np.logical_and(xlog < np.max(logxdat),xlog > np.min(logxdat))] # Interp only in the relevant interval
            N1 = np.power(10,1.*np.interp(xlogin, logxdat, logydat))
            N1[N1 < np.min(N1)*0.95] = 0
            
            Ntmp[np.logical_and(xlog < np.max(logxdat),xlog > np.min(logxdat))]=N1
#             print("In summing",Ntmp)
            NFull=NFull+Ntmp
    else: # One dimensional array case
      
        ydatA[ydatA == 0] = 1e-40
        logydat=np.log(ydatA)/np.log(10.)
        logxdat=np.log(xdatA)/np.log(10.)
        N1 = np.power(10,1.*np.interp(xlog, logxdat, logydat))
        N1[N1 < np.min(N1)*0.95] = 0
        NFull=NFull+N1

    return(np.power(10,1.*xlog),NFull)


### ---- Combines two approximation and interpolate in between
def CombineAndInterp(f1,f2,xlow,xhigh,x):


    Res1=(x<xlow)*np.nan_to_num(f1(x))+(x>xhigh)*np.nan_to_num(f2(x))
    ResmiddleLog=(x>xlow)*(x<xhigh)*np.logical_and(f2(xhigh)>0.,f1(xlow)>0.)*np.exp( np.log(f2(xhigh)) + np.log(x/xhigh)/np.log(xlow/xhigh)*np.log(f1(xlow)/f2(xhigh)) )
    ResmiddleLin = (x>xlow)*(x<xhigh)*np.logical_or(f2(xhigh)==0.,f1(xlow)==0.)*(f1(xlow)+(x-xlow)/(xhigh-xlow)*(f2(xhigh)-f1(xlow))) # Used only if one of the term is zero
    
    return (Res1+np.nan_to_num(ResmiddleLog)+np.nan_to_num(ResmiddleLin))


##########################################3##################
############# Other useful functions        #################3
##########################################3##################

#### Give the proper integration range depending on the experiments
def GetxiRange(channel,delrec):
    npoint=8
    MJPsi = 3.097;MCapitalUpsilon = 9.460; # 1S resonance for   Upsilon*)
    MPhi = 1.019;MBminus = 5.27931; MB0 = 5.36682 ;MKminus = 0.493677 ;Mpim = 0.1396;MK0 = 0.497648
    MPi = 0.1349766; MEta = 0.547862; MEtap = 0.9577
    MRho = 0.77526; MOmega = 0.78265;MDs = 2.00696; MD = 1.8648;MKs = 0.89581;MK = 0.49761  
    M2toM12=(2+delrec)/(1+delrec)
   
    if channel == "babar_BmtoKmnunu":
        xhigh=(MBminus-MKminus)/M2toM12
    elif channel == "belle2_BmtoKmnunu":
        xhigh=(MBminus-MKminus)/M2toM12
    elif channel == "belle_B0toPi0nunu":
        xhigh=(MB0-MPi)/M2toM12
    elif channel == "belle_B0toK0nunu":
        xhigh=(MB0-MK0)/M2toM12
    elif channel == "babar_BmtoPimnunu":
        xhigh=(MBminus- Mpim)/M2toM12
    elif channel == "e391a_KL0toPi0nunu" or channel == "na62_KL0toPi0nunu":
        xhigh=(MK0-MPi)/M2toM12
    elif channel == "e949_KptoPipa":
        xhigh=(MKminus-Mpim)/M2toM12
    elif channel == "na62_KptoPipa":
        xhigh=(MKminus-Mpim)/M2toM12
    elif channel == "na62_pi0toinvisible":
        xhigh=(MPi)/M2toM12
    elif channel == "bes_JPsitoinvisible":
        xhigh=(MJPsi)/M2toM12
    elif channel == "babar_Upsilontoinvisible":
        xhigh=(MCapitalUpsilon)/M2toM12
    else:
        xhigh=10.
    
    # We sample more the threshold since calculating points can be costly
    
    if 0.4*xhigh>0.001:
        xi1=log_sample(0.001, 0.4*xhigh, npoint)
        xi2=log_sample( 0.45*xhigh, 0.99*xhigh, npoint)
        xi=np.concatenate((xi1, xi2), axis=0)
    else:
        xi=log_sample(0.001, 0.99*xhigh, 2*npoint)
    return xi
 


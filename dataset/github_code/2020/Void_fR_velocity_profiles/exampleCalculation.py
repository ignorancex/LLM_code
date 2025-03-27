#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:34:17 2020

@author: christopherwilson
"""

from voidAnalysisMethods import *

voidDir="path/to/directory/containing/voidCatalog/"
voidDir="/Users/christopherwilson/Documents/GitHub/Void_fR_velocity_profiles/voidData/"



"Load in voids here"
GR_partVoidPath_z0p0=voidDir+"GR_partVoids_z0p0.npy"
GR_partVoids_z0p0=np.load(GR_partVoidPath_z0p0,allow_pickle=True,encoding="latin1")
F6_partVoidPath_z0p0=voidDir+"F6_partVoids_z0p0.npy"
F6_partVoids_z0p0=np.load(F6_partVoidPath_z0p0,allow_pickle=True,encoding="latin1")
F5_partVoidPath_z0p0=voidDir+"F5_partVoids_z0p0.npy"
F5_partVoids_z0p0=np.load(F5_partVoidPath_z0p0,allow_pickle=True,encoding="latin1")

GR_haloVoidPath_z0p0=voidDir+"GR_haloVoids_z0p0.npy"
GR_haloVoids_z0p0=np.load(GR_haloVoidPath_z0p0,allow_pickle=True,encoding="latin1")
F6_haloVoidPath_z0p0=voidDir+"F6_haloVoids_z0p0.npy"
F6_haloVoids_z0p0=np.load(F6_haloVoidPath_z0p0,allow_pickle=True,encoding="latin1")
F5_haloVoidPath_z0p0=voidDir+"F5_haloVoids_z0p0.npy"
F5_haloVoids_z0p0=np.load(F5_haloVoidPath_z0p0,allow_pickle=True,encoding="latin1")

GR_partVoidPath_z0p5=voidDir+"GR_partVoids_z0p5.npy"
GR_partVoids_z0p5=np.load(GR_partVoidPath_z0p5,allow_pickle=True,encoding="latin1")
F6_partVoidPath_z0p5=voidDir+"F6_partVoids_z0p5.npy"
F6_partVoids_z0p5=np.load(F6_partVoidPath_z0p5,allow_pickle=True,encoding="latin1")
F5_partVoidPath_z0p5=voidDir+"F5_partVoids_z0p5.npy"
F5_partVoids_z0p5=np.load(F5_partVoidPath_z0p5,allow_pickle=True,encoding="latin1")

GR_haloVoidPath_z0p5=voidDir+"GR_haloVoids_z0p5.npy"
GR_haloVoids_z0p5=np.load(GR_haloVoidPath_z0p5,allow_pickle=True,encoding="latin1")
F6_haloVoidPath_z0p5=voidDir+"F6_haloVoids_z0p5.npy"
F6_haloVoids_z0p5=np.load(F6_haloVoidPath_z0p5,allow_pickle=True,encoding="latin1")
F5_haloVoidPath_z0p5=voidDir+"F5_haloVoids_z0p5.npy"
F5_haloVoids_z0p5=np.load(F5_haloVoidPath_z0p5,allow_pickle=True,encoding="latin1")



"commented out below is how to re-classify voids. you can change the Rcut values or the deltaCut values to adjust what is considered R type or S type."
"If you want to re-classify voids based on the particle data, have to set 'fromParts=True'" 

# reclassifyVoidsRS(GR_haloVoids_z0p5,Rcut=1.0)
# reclassifyVoidsRS(F6_haloVoids_z0p5,Rcut=1.0)
# reclassifyVoidsRS(F5_haloVoids_z0p5,Rcut=1.0)
# matchClassificatons(GR_haloVoids_z0p5,GR_partVoids_z0p5)
# matchClassificatons(F6_haloVoids_z0p5,F6_partVoids_z0p5)
# matchClassificatons(F5_haloVoids_z0p5,F5_partVoids_z0p5)

# reclassifyVoidsRS(GR_haloVoids_z0p0,Rcut=1.0)
# reclassifyVoidsRS(F6_haloVoids_z0p0,Rcut=1.0)
# reclassifyVoidsRS(F5_haloVoids_z0p0,Rcut=1.0)
# matchClassificatons(GR_haloVoids_z0p0,GR_partVoids_z0p0)
# matchClassificatons(F6_haloVoids_z0p0,F6_partVoids_z0p0)
# matchClassificatons(F5_haloVoids_z0p0,F5_partVoids_z0p0)

########################################################################################################################

"calculate density profile for GR, F5, and F6 from particles, for 'R' type voids at z=0, with R_eff between 5 and 15 Mpc/h"
rMin=5
rMax=15

densProfGR=buildDensityProfile(GR_partVoids_z0p0,voidType=["R"],rMin=rMin,rMax=rMax,weight=0,outTo=50) 
densProfF6=buildDensityProfile(F6_partVoids_z0p0,voidType=["R"],rMin=rMin,rMax=rMax,weight=0,outTo=50) 
densProfF5=buildDensityProfile(F5_partVoids_z0p0,voidType=["R"],rMin=rMin,rMax=rMax,weight=0,outTo=50) 

"Here, the 'outTo' option tells the profiles to be constructed using all 50 bins, so the profile ranges from 0 to 5R_eff in step sizes of 0.1 Reff."
"'partVoid' profiles have density information out to 5 Reff, so the max value of 'outTo' for them is 50. for 'haloVoids' the max value 30 as the data only goes to 3 Reff"

"'weight' is used to weight the density profiles by certain powers of their effective radius. It is primarily used when calculating newtonian forces with the 'calcGreensNewtonForceWithError' method."
"Setting weight=w would produce the average value of delta*Reff^w so where profile is weighted by its own effective radius to the power of w"


plt.errorbar(densProfGR[0],densProfGR[1],densProfGR[2])
plt.errorbar(densProfF6[0],densProfF6[1],densProfF6[2])
plt.errorbar(densProfF5[0],densProfF5[1],densProfF5[2])

plt.clf()

#########################################################################################################################

"calculate velocity profiles from the particles for the same set of voids, first unintegrated"
velProfGR=buildVelocityProfile(GR_partVoids_z0p0,voidType=["R"],catType='part',rMin=rMin,rMax=rMax,weight=0)
velProfF6=buildVelocityProfile(F6_partVoids_z0p0,voidType=["R"],catType='part',rMin=rMin,rMax=rMax,weight=0)
velProfF5=buildVelocityProfile(F5_partVoids_z0p0,voidType=["R"],catType='part',rMin=rMin,rMax=rMax,weight=0)

plt.errorbar(velProfGR[0],velProfGR[1],velProfGR[2])
plt.errorbar(velProfF6[0],velProfF6[1],velProfF6[2])
plt.errorbar(velProfF5[0],velProfF5[1],velProfF5[2])

plt.ylabel("km/s")
plt.xlabel("r/R_eff")


plt.clf()


#########################################################################################################################

"calculate velocity profiles from the halos for the same set of voids, now integrated"
velProfGR=buildIntegratedVelocityProfile(GR_haloVoids_z0p0,voidType=["R"],catType='halo',rMin=rMin,rMax=rMax,weight=0,minFromCenter=2.5)
velProfF6=buildIntegratedVelocityProfile(F6_haloVoids_z0p0,voidType=["R"],catType='halo',rMin=rMin,rMax=rMax,weight=0,minFromCenter=2.5)
velProfF5=buildIntegratedVelocityProfile(F5_haloVoids_z0p0,voidType=["R"],catType='halo',rMin=rMin,rMax=rMax,weight=0,minFromCenter=2.5)

"minFromCenter excluses those tracers from the average who are closer than x Mps/h from the void center, this helps minimuze error with regard to the void center being slightly misplaced, although it has minimal effect in the later bins "



plt.errorbar(velProfGR[0],velProfGR[1],velProfGR[2])
plt.errorbar(velProfF6[0],velProfF6[1],velProfF6[2])
plt.errorbar(velProfF5[0],velProfF5[1],velProfF5[2])

plt.ylabel("km/s")
plt.xlabel("r/R_eff")

plt.clf()


#########################################################################################################################


"calculate the newtonian force for these voids in GR"

"instead of calculating the newtonian force for each void individually, we can weight each profile by its effective radius and use this weighted density profile as the source term instead"
weightedDensProfGR=buildDensityProfile(GR_partVoids_z0p0,voidType=["R"],rMin=rMin,rMax=rMax,weight=1,outTo=50) # notice how weight is set to 1 this time, so we are averageing delta*R_{eff}^1
newtonForceGR=calcGreensNewtonForceWithError(weightedDensProfGR,z=0,Limit=10000,dx=0.00001,outTo=5,omega_M=0.281)

plt.plot(newtonForceGR[0][0:150],newtonForceGR[1][0:150],linestyle='-',linewidth=1)
plt.fill_between(newtonForceGR[0][0:150], newtonForceGR[1][0:150]-newtonForceGR[2][0:150], newtonForceGR[1][0:150]+newtonForceGR[2][0:150],alpha=0.5)


"we can also calculate the newtonian force for the average profile using the average Reff. Dont get errors this way"
ReffAvgGR=getAvgReff(GR_partVoids_z0p0,voidType=["R"],rMin=5,rMax=15)

newtonForceGR2=calcGreensNewtonForce(densProfGR,ReffAvgGR,z=0,Limit=10000,dx=0.00001,outTo=5,omega_M=0.281,omega_L=0.719)
plt.plot(newtonForceGR2[0][0:150],newtonForceGR2[1][0:150])

plt.ylabel("H0*c")
plt.xlabel("r/R_eff")

"We find decent agreement between the newtonian forces calculated in this manner"

plt.clf()

#########################################################################################################################


"We can calculate the linearized fifth force and the full fifth force for the average profile in F6 as follows"
densProfF6=buildDensityProfile(F6_partVoids_z0p0,voidType=["R"],rMin=5,rMax=15,weight=0,outTo=50) 
ReffAvgF6=getAvgReff(F6_partVoids_z0p0,voidType=["R"],rMin=5,rMax=15)
linFifthF6=calcGreensFifthForce(densProfF6,ReffAvgF6,F0=1e-6,z=0,Limit=10000,dx=0.00001,outTo=5,omega_M=0.281,omega_L=0.719)
plt.plot(linFifthF6[0],linFifthF6[1])



fullFifthF6,alphas,newDens=convergeForce(densProfF6,ReffAvgF6,z=0,dampFactor=0.75,maxLoops=10,binsToIgnore=10,tolerance=0.0075,outTo=5,F0=1e-6,fracBeforeMod=0.9,omega_M=0.281,omega_L=0.719)
plt.plot(fullFifthF6[0],fullFifthF6[1])

"when running the various methods related to forces, 'outTo' now refers to how many Reff you want to calculate the force out to. the max value is 5"
"outTo=5 will include the entirety of the density profile from 'partVoids' and is therefor recommended, although not necessary"


plt.ylabel("H0*c")
plt.xlabel("r/R_eff")

plt.clf()

#########################################################################################################################

"We have ran this 'convergeForce' for each individual void density profile, and saved the results in the following numpy object arrays"

"load Forces in here"

"The first time this is run, you will have to load in each realization individually due to file size contraints on github using the first block of code provided, and then run the second block to load in the single file containing all 5 realizations"
"After the first time, you can run the second block of code"

"if True: added for code folding purposes and to easily turn on / off"


"Block 1"
if True:
    F5_fifthConverged_real1Path_z0p0=voidDir+"F5_fifthConverged_real1_z0p0.npy"
    F5_fifthConverged_real1_z0p0=np.load(F5_fifthConverged_real1Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real2Path_z0p0=voidDir+"F5_fifthConverged_real2_z0p0.npy"
    F5_fifthConverged_real2_z0p0=np.load(F5_fifthConverged_real2Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real3Path_z0p0=voidDir+"F5_fifthConverged_real3_z0p0.npy"
    F5_fifthConverged_real3_z0p0=np.load(F5_fifthConverged_real3Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real4Path_z0p0=voidDir+"F5_fifthConverged_real4_z0p0.npy"
    F5_fifthConverged_real4_z0p0=np.load(F5_fifthConverged_real4Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real5Path_z0p0=voidDir+"F5_fifthConverged_real5_z0p0.npy"
    F5_fifthConverged_real5_z0p0=np.load(F5_fifthConverged_real5Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real1Path_z0p0=voidDir+"F6_fifthConverged_real1_z0p0.npy"
    F6_fifthConverged_real1_z0p0=np.load(F6_fifthConverged_real1Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real2Path_z0p0=voidDir+"F6_fifthConverged_real2_z0p0.npy"
    F6_fifthConverged_real2_z0p0=np.load(F6_fifthConverged_real2Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real3Path_z0p0=voidDir+"F6_fifthConverged_real3_z0p0.npy"
    F6_fifthConverged_real3_z0p0=np.load(F6_fifthConverged_real3Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real4Path_z0p0=voidDir+"F6_fifthConverged_real4_z0p0.npy"
    F6_fifthConverged_real4_z0p0=np.load(F6_fifthConverged_real4Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real5Path_z0p0=voidDir+"F6_fifthConverged_real5_z0p0.npy"
    F6_fifthConverged_real5_z0p0=np.load(F6_fifthConverged_real5Path_z0p0,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real1Path_z0p5=voidDir+"F5_fifthConverged_real1_z0p5.npy"
    F5_fifthConverged_real1_z0p5=np.load(F5_fifthConverged_real1Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real2Path_z0p5=voidDir+"F5_fifthConverged_real2_z0p5.npy"
    F5_fifthConverged_real2_z0p5=np.load(F5_fifthConverged_real2Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real3Path_z0p5=voidDir+"F5_fifthConverged_real3_z0p5.npy"
    F5_fifthConverged_real3_z0p5=np.load(F5_fifthConverged_real3Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real4Path_z0p5=voidDir+"F5_fifthConverged_real4_z0p5.npy"
    F5_fifthConverged_real4_z0p5=np.load(F5_fifthConverged_real4Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F5_fifthConverged_real5Path_z0p5=voidDir+"F5_fifthConverged_real5_z0p5.npy"
    F5_fifthConverged_real5_z0p5=np.load(F5_fifthConverged_real5Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real1Path_z0p5=voidDir+"F6_fifthConverged_real1_z0p5.npy"
    F6_fifthConverged_real1_z0p5=np.load(F6_fifthConverged_real1Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real2Path_z0p5=voidDir+"F6_fifthConverged_real2_z0p5.npy"
    F6_fifthConverged_real2_z0p5=np.load(F6_fifthConverged_real2Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real3Path_z0p5=voidDir+"F6_fifthConverged_real3_z0p5.npy"
    F6_fifthConverged_real3_z0p5=np.load(F6_fifthConverged_real3Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real4Path_z0p5=voidDir+"F6_fifthConverged_real4_z0p5.npy"
    F6_fifthConverged_real4_z0p5=np.load(F6_fifthConverged_real4Path_z0p5,allow_pickle=True,encoding="latin1")
    
    F6_fifthConverged_real5Path_z0p5=voidDir+"F6_fifthConverged_real5_z0p5.npy"
    F6_fifthConverged_real5_z0p5=np.load(F6_fifthConverged_real5Path_z0p5,allow_pickle=True,encoding="latin1")
    
    
    
    
    
    "F5, z=0"
    size=len(F5_fifthConverged_real1_z0p0)+len(F5_fifthConverged_real2_z0p0)+len(F5_fifthConverged_real3_z0p0)+len(F5_fifthConverged_real4_z0p0)+len(F5_fifthConverged_real5_z0p0)
    F=np.zeros([size,2,250])
    fifthArray=np.array([F5_fifthConverged_real1_z0p0,F5_fifthConverged_real2_z0p0,F5_fifthConverged_real3_z0p0,F5_fifthConverged_real4_z0p0,F5_fifthConverged_real5_z0p0])
    q=0
    for i in range(len(fifthArray)):
        for j in range(len(fifthArray[i])):
            F[q][0]=fifthArray[i][j][0]
            F[q][1]=fifthArray[i][j][1]
            q+=1
    np.save(voidDir+"F5_fifthConverged_z0p0",F)
    Q=np.load(voidDir+"F5_fifthConverged_z0p0.npy")
    
    
    
    
    
    "F6, z=0"
    size=len(F6_fifthConverged_real1_z0p0)+len(F6_fifthConverged_real2_z0p0)+len(F6_fifthConverged_real3_z0p0)+len(F6_fifthConverged_real4_z0p0)+len(F6_fifthConverged_real5_z0p0)
    F=np.zeros([size,2,250])
    fifthArray=np.array([F6_fifthConverged_real1_z0p0,F6_fifthConverged_real2_z0p0,F6_fifthConverged_real3_z0p0,F6_fifthConverged_real4_z0p0,F6_fifthConverged_real5_z0p0])
    q=0
    for i in range(len(fifthArray)):
        for j in range(len(fifthArray[i])):
            F[q][0]=fifthArray[i][j][0]
            F[q][1]=fifthArray[i][j][1]
            q+=1
    np.save(voidDir+"F6_fifthConverged_z0p0",F)
    Q=np.load(voidDir+"F6_fifthConverged_z0p0.npy")


    "F5, z=0.5"
    size=len(F5_fifthConverged_real1_z0p5)+len(F5_fifthConverged_real2_z0p5)+len(F5_fifthConverged_real3_z0p5)+len(F5_fifthConverged_real4_z0p5)+len(F5_fifthConverged_real5_z0p5)
    F=np.zeros([size,2,250])
    fifthArray=np.array([F5_fifthConverged_real1_z0p5,F5_fifthConverged_real2_z0p5,F5_fifthConverged_real3_z0p5,F5_fifthConverged_real4_z0p5,F5_fifthConverged_real5_z0p5])
    q=0
    for i in range(len(fifthArray)):
        for j in range(len(fifthArray[i])):
            F[q][0]=fifthArray[i][j][0]
            F[q][1]=fifthArray[i][j][1]
            q+=1
    np.save(voidDir+"F5_fifthConverged_z0p5",F)
    Q=np.load(voidDir+"F5_fifthConverged_z0p5.npy")
    
    
    
    
    
    "F6, z=0.5"
    size=len(F6_fifthConverged_real1_z0p5)+len(F6_fifthConverged_real2_z0p5)+len(F6_fifthConverged_real3_z0p5)+len(F6_fifthConverged_real4_z0p5)+len(F6_fifthConverged_real5_z0p5)
    F=np.zeros([size,2,250])
    fifthArray=np.array([F6_fifthConverged_real1_z0p5,F6_fifthConverged_real2_z0p5,F6_fifthConverged_real3_z0p5,F6_fifthConverged_real4_z0p5,F6_fifthConverged_real5_z0p5])
    q=0
    for i in range(len(fifthArray)):
        for j in range(len(fifthArray[i])):
            F[q][0]=fifthArray[i][j][0]
            F[q][1]=fifthArray[i][j][1]
            q+=1
    np.save(voidDir+"F6_fifthConverged_z0p5",F)
    Q=np.load(voidDir+"F6_fifthConverged_z0p5.npy")






"Block 2"
if True:
    F6_fifthConvergedPath_z0p0=voidDir+"F6_fifthConverged_z0p0.npy"
    F6_fifthConverged_z0p0=np.load(F6_fifthConvergedPath_z0p0,allow_pickle=True,encoding="latin1")
    F5_fifthConvergedPath_z0p0=voidDir+"F5_fifthConverged_z0p0.npy"
    F5_fifthConverged_z0p0=np.load(F5_fifthConvergedPath_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConvergedPath_z0p5=voidDir+"F6_fifthConverged_z0p5.npy"
    F6_fifthConverged_z0p5=np.load(F6_fifthConvergedPath_z0p5,allow_pickle=True,encoding="latin1")
    F5_fifthConvergedPath_z0p5=voidDir+"F5_fifthConverged_z0p5.npy"
    F5_fifthConverged_z0p5=np.load(F5_fifthConvergedPath_z0p5,allow_pickle=True,encoding="latin1")
    
    
    F6_fifthConvergedPath_z0p0=voidDir+"F6_fifthConverged_z0p0.npy"
    F6_fifthConverged_z0p0=np.load(F6_fifthConvergedPath_z0p0,allow_pickle=True,encoding="latin1")
    F5_fifthConvergedPath_z0p0=voidDir+"F5_fifthConverged_z0p0.npy"
    F5_fifthConverged_z0p0=np.load(F5_fifthConvergedPath_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_fifthConvergedPath_z0p5=voidDir+"F6_fifthConverged_z0p5.npy"
    F6_fifthConverged_z0p5=np.load(F6_fifthConvergedPath_z0p5,allow_pickle=True,encoding="latin1")
    F5_fifthConvergedPath_z0p5=voidDir+"F5_fifthConverged_z0p5.npy"
    F5_fifthConverged_z0p5=np.load(F5_fifthConvergedPath_z0p5,allow_pickle=True,encoding="latin1")

    "We also saved the values of alpha and the max error between the real and re-constructed density profile from the algorithm for each void"
    F6_alphaErrorsPath_z0p0=voidDir+"F6_alphaErrors_z0p0.npy"
    F6_alphaErrors_z0p0=np.load(F6_alphaErrorsPath_z0p0,allow_pickle=True,encoding="latin1")
    F5_alphaErrorsPath_z0p0=voidDir+"F5_alphaErrors_z0p0.npy"
    F5_alphaErrors_z0p0=np.load(F5_alphaErrorsPath_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_alphaErrorsPath_z0p5=voidDir+"F6_alphaErrors_z0p5.npy"
    F6_alphaErrors_z0p5=np.load(F6_alphaErrorsPath_z0p5,allow_pickle=True,encoding="latin1")
    F5_alphaErrorsPath_z0p5=voidDir+"F5_alphaErrors_z0p5.npy"
    F5_alphaErrors_z0p5=np.load(F5_alphaErrorsPath_z0p5,allow_pickle=True,encoding="latin1")
    
    
    F6_alphaErrorsPath_z0p0=voidDir+"F6_alphaErrors_z0p0.npy"
    F6_alphaErrors_z0p0=np.load(F6_alphaErrorsPath_z0p0,allow_pickle=True,encoding="latin1")
    F5_alphaErrorsPath_z0p0=voidDir+"F5_alphaErrors_z0p0.npy"
    F5_alphaErrors_z0p0=np.load(F5_alphaErrorsPath_z0p0,allow_pickle=True,encoding="latin1")
    
    F6_alphaErrorsPath_z0p5=voidDir+"F6_alphaErrors_z0p5.npy"
    F6_alphaErrors_z0p5=np.load(F6_alphaErrorsPath_z0p5,allow_pickle=True,encoding="latin1")
    F5_alphaErrorsPath_z0p5=voidDir+"F5_alphaErrors_z0p5.npy"
    F5_alphaErrors_z0p5=np.load(F5_alphaErrorsPath_z0p5,allow_pickle=True,encoding="latin1")

#########################################################################################################################


"Since we now have the 5th force for every void individually, we can now get errors on our 5th force by averaging the fifth force computed in each void indivudually"

fullFifthF6_withErrors=filterFifthForce(F6_fifthConverged_z0p0,F6_partVoids_z0p0,rMin=5,rMax=15,alphaCat=F6_alphaErrors_z0p0,errorCut=0.05,voidType=["R"])
plt.plot(fullFifthF6_withErrors[0],fullFifthF6_withErrors[1],linestyle='-',linewidth=1)
plt.fill_between(fullFifthF6_withErrors[0], fullFifthF6_withErrors[1]-fullFifthF6_withErrors[2], fullFifthF6_withErrors[1]+fullFifthF6_withErrors[2],alpha=0.5)

plt.plot(fullFifthF6[0],fullFifthF6[1])
plt.ylabel("H0*c")
plt.xlabel("r/R_eff")

"We can see excellent agreement between the fifth force calculated in each void individually and then averaged versus that resulting from averaging the density profiles and then computing the fifth force"

"I make no claims that the methods presented here are optimized for speed. If any particular user has comments on particular methods for improvement, they are welcome "






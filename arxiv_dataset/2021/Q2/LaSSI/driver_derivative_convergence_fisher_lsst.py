import cmb
reload(cmb)
from cmb import *

import cmb_lensing_rec
reload(cmb_lensing_rec)
from cmb_lensing_rec import *

import fisher_lsst
reload(fisher_lsst)
from fisher_lsst import *

##################################################################################
# Forecast parameters

## for debugging
#nBins = 3  # 10
#nL = 10  # 50, 20, 100
#fsky = 0.4

# actual params
nBins =  10  
nL = 50  
fsky = 0.35


# cosmological parameters to include
massiveNu = True  
wCDM = True 
curvature = True 

# priors to include
PlanckPrior = True

# include a known magnification bias
magBias = True


# Parallel evaluations
nProc = 4   # not actually used

##################################################################################
# CMB lensing noise

# CMB S4
cmb = CMB(beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=3.e3, lMaxP=5.e3, atm=False, name="cmbs4")
cmbLensRec = CMBLensRec(cmb, save=False, nProc=nProc)
fNk = cmbLensRec.fN_k_mv


##################################################################################
# Fisher calculation

import fisher_lsst
reload(fisher_lsst)
from fisher_lsst import *

fsky = 0.35 # 0.4

##################################################################################

save = True

#derivStepSizes = np.array([0.01, 0.05, 0.1, 0.5, 1., 1.5, 2.])
derivStepSizes = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 1.5, 2.])#, 1.75]
#derivStepSizes = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.])#, 1.5, 2.])#, 1.75]
#derivStepSizes = np.array([2.])
nStep = len(derivStepSizes)

iStepFid = 6#-1 #4


nData = 11550  # read from the output of the fisher class
nPar = 140  # read from the output of the fisher class
# data vector derivatives
deriv  = np.zeros((nStep, nPar, nData))
# Posterior uncertainties
par = np.empty(nStep, dtype=object)
s = np.zeros((nStep, nPar))



for iStep in range(nStep):
   derivStepSize = derivStepSizes[iStep]
   # forecast name
   name = 'derivstepsize'+floatExpForm(derivStepSize, round=2)

   # Parameter classes
   cosmoPar = CosmoParams(massiveNu=massiveNu, wCDM=wCDM, curvature=curvature, PlanckPrior=PlanckPrior, derivStepSize=derivStepSize)
   #print 'cosmo par'
   #print cosmoPar.high - cosmoPar.low
   galaxyBiasPar = GalaxyBiasParams(nBins=nBins, derivStepSize=derivStepSize)
   #print 'gal bias'
   #print galaxyBiasPar.high - galaxyBiasPar.low
   shearMultBiasPar = ShearMultBiasParams(nBins=nBins, derivStepSize=derivStepSize)
   #print 'shear mult bias'
   #print shearMultBiasPar.high - shearMultBiasPar.low
   photoZPar = PhotoZParams(nBins=nBins, outliers=0.1, derivStepSize=derivStepSize)
   #print 'photo z'
   #print photoZPar.high - photoZPar.low

   # Perform the Fisher forecast
   fish = FisherLsst(cosmoPar, galaxyBiasPar, shearMultBiasPar, photoZPar, nBins=nBins, nL=nL, fsky=fsky, fNk=fNk, magBias=magBias, name=name, nProc=nProc, save=save)

   # read the derivatives
   deriv[iStep,:,:] = fish.derivativeDataVector
   
   # extract the posterior uncertainties
   par[iStep], s[iStep,:] = fish.computePosterior()


##################################################################################
# Fiducial analysis, just to get the figure path...

cosmoPar = CosmoParams(massiveNu=massiveNu, wCDM=wCDM, curvature=curvature, PlanckPrior=PlanckPrior)
galaxyBiasPar = GalaxyBiasParams(nBins=nBins)
shearMultBiasPar = ShearMultBiasParams(nBins=nBins)
photoZPar = PhotoZParams(nBins=nBins, outliers=0.1)

fish = FisherLsst(cosmoPar, galaxyBiasPar, shearMultBiasPar, photoZPar, nBins=nBins, nL=nL, fsky=fsky, fNk=fNk, magBias=magBias, name=None, nProc=nProc, save=False)



##################################################################################
##################################################################################

##################################################################################
##################################################################################
# Check diagonal Fisher matrix


diagFisher = np.zeros((nStep, nPar))
for iStep in range(nStep):
   for iPar in range(nPar):
      diagFisher[iStep, iPar] = par[iStep].fisher[iPar, iPar]

diagFisherRelDiff = np.sqrt(diagFisher[:,:]) / np.sqrt(diagFisher[iStepFid,:][np.newaxis,:]) - 1.


fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(cosmoPar.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  diagFisherRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  diagFisherRelDiff[:,iPar], label=par[0].namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(cosmoPar.nPar))
ax.set_yticklabels(par[0].namesLatex[range(cosmoPar.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (cosmoPar.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{unmarg} = \left(F_{\alpha, \alpha}\right)^{-1/2}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_unmarg.pdf', bbox_inches='tight')

plt.show()


##################################################################################
##################################################################################
##################################################################################
# Check marginalized cosmo param constraints


##################################################################################
# Jointly fit for LCDM + neutrinos + w0,wa + curvature

sRelDiff = s[:,:] / s[iStepFid,:][np.newaxis,:] - 1.


fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(cosmoPar.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  sRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  sRelDiff[:,iPar], label=par[0].namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(cosmoPar.nPar))
ax.set_yticklabels(par[0].namesLatex[range(cosmoPar.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (cosmoPar.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{marg} = F^{-1}_{\alpha, \alpha}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_marg_allatonce_lcdmw0wamnucurv.pdf', bbox_inches='tight')

plt.show()


##################################################################################
# Jointly fit for LCDM + neutrinos

I = cosmoPar.ILCDMMnu + range(cosmoPar.nPar,nPar)
s2 = np.zeros((nStep, len(cosmoPar.ILCDMMnu)))
for iStep in range(nStep):
   par2 = par[iStep].extractParams(I, marg=False)
   par2 = par2.extractParams(range(len(cosmoPar.ILCDMMnu)), marg=True)
   s2[iStep,:] = par2.paramUncertainties(marg=True)

sRelDiff = s2[:,:] / s2[iStepFid,:][np.newaxis,:] - 1.


fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(par2.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  sRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  sRelDiff[:,iPar], label=par2.namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(par2.nPar))
ax.set_yticklabels(par2.namesLatex[range(par2.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (par2.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{marg} = F^{-1}_{\alpha, \alpha}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_marg_allatonce_lcdmmnu.pdf', bbox_inches='tight')

plt.show()


##################################################################################
# Jointly fit for LCDM + w0, wa

I = cosmoPar.ILCDMW0Wa + range(cosmoPar.nPar,nPar)
s2 = np.zeros((nStep, len(cosmoPar.ILCDMW0Wa)))
for iStep in range(nStep):
   par2 = par[iStep].extractParams(I, marg=False)
   par2 = par2.extractParams(range(len(cosmoPar.ILCDMW0Wa)), marg=True)
   s2[iStep,:] = par2.paramUncertainties(marg=True)

sRelDiff = s2[:,:] / s2[iStepFid,:][np.newaxis,:] - 1.


fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(par2.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  sRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  sRelDiff[:,iPar], label=par2.namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(par2.nPar))
ax.set_yticklabels(par2.namesLatex[range(par2.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (par2.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{marg} = F^{-1}_{\alpha, \alpha}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_marg_allatonce_lcdmw0wa.pdf', bbox_inches='tight')

plt.show()


##################################################################################
# Jointly fit for LCDM + curvature

I = cosmoPar.ILCDMCurv + range(cosmoPar.nPar,nPar)
s2 = np.zeros((nStep, len(cosmoPar.ILCDMCurv)))
for iStep in range(nStep):
   par2 = par[iStep].extractParams(I, marg=False)
   par2 = par2.extractParams(range(len(cosmoPar.ILCDMCurv)), marg=True)
   s2[iStep,:] = par2.paramUncertainties(marg=True)

sRelDiff = s2[:,:] / s2[iStepFid,:][np.newaxis,:] - 1.


fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(par2.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  sRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  sRelDiff[:,iPar], label=par2.namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(par2.nPar))
ax.set_yticklabels(par2.namesLatex[range(par2.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (par2.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{marg} = F^{-1}_{\alpha, \alpha}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_marg_allatonce_lcdmcurv.pdf', bbox_inches='tight')

plt.show()






##################################################################################



##################################################################################
##################################################################################
##################################################################################
##################################################################################
# For the marginalized posteriors, vary only the corresponding parameter step,
# not all of them at once

# first param is the one we will forecast
derivMixed  = np.zeros((nPar, nStep, nPar, nData))
fisherData  = np.zeros((nPar, nStep, nPar, nPar))
sMixed = np.zeros((nPar, nStep, nPar))

for iPar in range(cosmoPar.nPar):
   print("Computing parameter "+str(iPar))
   for iStep in range(nStep):

      # generate the derivative where only the iPar par is rescaled
      derivMixed[iPar,iStep,:,:] = fish.derivativeDataVector.copy()
      derivMixed[iPar,iStep,iPar,:] = deriv[iStep,iPar,:]

      # get the corresponding Fisher
      fisherData[iPar,iStep,:,:] = fish.generateFisher(mask=None, deriv=derivMixed[iPar,iStep,:,:])

      # get the posterior constraints
      _, sMixed[iPar,iStep,:] = fish.computePosterior(fisherData=fisherData[iPar,iStep,:,:])



# Get the relative changes
sMixedRelDiff = np.zeros((nStep, nPar))
for iPar in range(cosmoPar.nPar):
   for iStep in range(nStep):
      sMixedRelDiff[iStep, iPar] = sMixed[iPar, iStep, iPar] / sMixed[iPar, iStepFid, iPar] - 1



fig=plt.figure(0, figsize=(8,6))
ax=fig.add_subplot(111)
#
for iPar in range(cosmoPar.nPar):
   ax.fill_between(derivStepSizes, 0.25*iPar - 0.1*np.ones(nStep), 0.25*iPar + 0.1*np.ones(nStep), facecolor='gray', edgecolor='', alpha=0.5)
   ax.axhline(0.25*iPar, color='gray', linestyle='--')
   #
   ax.plot(derivStepSizes[iStepFid], 0.25*iPar +  sMixedRelDiff[iStepFid,iPar], 'ko')
   ax.plot(derivStepSizes, 0.25*iPar +  sMixedRelDiff[:,iPar], label=par[0].namesLatex[iPar], alpha=1.)
#
ax.set_yticks(0.25*np.arange(cosmoPar.nPar))
ax.set_yticklabels(par[0].namesLatex[range(cosmoPar.nPar)])
ax.set_xlim((derivStepSizes[0], derivStepSizes[-1]))
ax.set_ylim((-0.25, (cosmoPar.nPar)*0.25))
#ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'Derivative step size')
ax.set_title(r'Relative change in $\sigma_\alpha^\text{marg} = F^{-1}_{\alpha, \alpha}$')
#
fig.savefig(fish.figurePath+'/convergence_uncertainty_marg_onebyone.pdf', bbox_inches='tight')

plt.show()



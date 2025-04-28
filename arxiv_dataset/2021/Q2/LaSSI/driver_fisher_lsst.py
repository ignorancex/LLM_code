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

# forecast name
name = None

# Parallel evaluations
nProc = 4   # not actually used

##################################################################################
# Parameter classes

cosmoPar = CosmoParams(massiveNu=massiveNu, wCDM=wCDM, curvature=curvature, PlanckPrior=PlanckPrior, derivStepSize=1.)
#cosmoPar.plotParams()
galaxyBiasPar = GalaxyBiasParams(nBins=nBins, derivStepSize=1.)
#galaxyBiasPar.plotParams()
shearMultBiasPar = ShearMultBiasParams(nBins=nBins, derivStepSize=1.)
#shearMultBiasPar = ShearMultBiasParams(nBins=nBins, mStd=1.e-5)   
#shearMultBiasPar.plotParams()

# Gaussian photo-z only:
#photoZPar = PhotoZParams(nBins=nBins)
# Photo-z with Gaussian core and outliers:
photoZPar = PhotoZParams(nBins=nBins, outliers=0.1, derivStepSize=1.)
#photoZPar.plotParams()

#cosmoPar.plotContours()


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

# same tomo bins for g and s
fish = FisherLsst(cosmoPar, galaxyBiasPar, shearMultBiasPar, photoZPar, nBins=nBins, nL=nL, fsky=fsky, fNk=fNk, magBias=magBias, name=name, nProc=nProc, save=False)


# different tomo bins for g and s
fishDiffgs = FisherLsst(cosmoPar, galaxyBiasPar, shearMultBiasPar, photoZPar, photoZSPar=photoZPar, nBins=nBins, nL=nL, fsky=fsky, fNk=fNk, magBias=magBias, name=name, nProc=nProc, save=False)


##################################################################################
# basic plots and checks



# Show observables and uncertainties
fish.plotEllBins(show=False)
fish.plotDndz(show=False)
fish.plotPowerSpectra(show=False)
fish.plotUncertaintyPowerSpectra(show=False)
fish.plotDiagCov()
#fish.plotCovMat(show=False)
#fish.plotInvCovMat(show=False)
fish.printSnrPowerSpectra()
fish.plotDerivativeDataVectorCosmo(show=False)
#fish.plotSingleDerivative("gg", 0, 0)
#fish.plotSingleDerivative("ss", 0, 15)
#fish.plotSingleDerivative("gg", 0, 20)


# outdated function, may not work
#fish.SampleDndz(photoZPar, nSamples=2)#, path=fish.figurePath+"/sampling_dndz_prior.pdf")


##################################################################################
##################################################################################
# Dependence on photo-z priors
# new version

fish.plotGPhotozRequirements(cosmoPar.ILCDMW0, name="lcdmw0", fish2=fishDiffgs)
fish.plotOutlierPhotozRequirements(cosmoPar.ILCDMW0, name="lcdmw0", fish2=fishDiffgs)

fish.plotGPhotozRequirements(cosmoPar.ILCDMW0Wa, name="lcdmw0wa", fish2=fishDiffgs)
fish.plotOutlierPhotozRequirements(cosmoPar.ILCDMW0Wa, name="lcdmw0wa", fish2=fishDiffgs)

fish.plotGPhotozRequirements(cosmoPar.ILCDMMnu, name="lcdmmnu", fish2=fishDiffgs)
fish.plotOutlierPhotozRequirements(cosmoPar.ILCDMMnu, name="lcdmmnu", fish2=fishDiffgs)

fish.plotGPhotozRequirements(cosmoPar.ILCDMCurv, name="lcdmcurv", fish2=fishDiffgs)
fish.plotOutlierPhotozRequirements(cosmoPar.ILCDMCurv, name="lcdmcurv", fish2=fishDiffgs)



##################################################################################
##################################################################################
# Comparison of cosmological parameters for the various runs

fish.plotSummaryComparison(ICosmoPar=cosmoPar.ILCDMW0Wa, name="lcdmw0wa")
fish.plotSummaryComparison(ICosmoPar=cosmoPar.ILCDMMnu, name="lcdmmnu")
fish.plotSummaryComparison(ICosmoPar=cosmoPar.ILCDMCurv, name="lcdmcurv")
fish.plotSummaryComparison(ICosmoPar=cosmoPar.ILCDMW0, name="lcdmw0")

fish.plotFomComparison(ICosmoPar=cosmoPar.ILCDMW0Wa, name="lcdmw0wa")
fish.plotFomComparison(ICosmoPar=cosmoPar.ILCDMW0, name="lcdmw0")


##################################################################################
##################################################################################
# Biases from wrong outlier photo-z

fish.plotBiasFromOutliers(cosmoPar.ILCDMW0, name='lcdmw0')
fish.plotBiasFromOutliers(cosmoPar.ILCDMW0Wa, name='lcdmw0wa')
fish.plotBiasFromOutliers(cosmoPar.ILCDMMnu, name='lcdmmnu')
fish.plotBiasFromOutliers(cosmoPar.ILCDMCurv, name='lcdmcurv')

##################################################################################
##################################################################################
# Contour plots


fishers=np.array([fish.fullPar.fisher, fish.fullPar.fisher+fish.fisherDataGs, fish.fullPar.fisher+fish.fisherDataGks])

# LCDM
fish.plotCosmoContours(cosmoPar.ILCDM, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=fish.figurePath+"/contours_lcdm.pdf")
# LCDMW0
fish.plotCosmoContours(cosmoPar.ILCDMW0, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=fish.figurePath+"/contours_lcdmw0.pdf")
# LCDMW0Wa
fish.plotCosmoContours(cosmoPar.ILCDMW0Wa, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=fish.figurePath+"/contours_lcdmw0wa.pdf")
# LCDMMnu
fish.plotCosmoContours(cosmoPar.ILCDMMnu, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=fish.figurePath+"/contours_lcdmmnu.pdf")
# LCDMCurv
fish.plotCosmoContours(cosmoPar.ILCDMCurv, fishers, fisherNames=['Planck', 'LSST', 'LSST + CMB lensing'], colors=['r', 'g', 'b'], path=fish.figurePath+"/contours_lcdmcurv.pdf")



# Examine Planck prior,
# to check degeneracy h0 - w in Planck prior
# LCDM
par = cosmoPar.extractParams(cosmoPar.ILCDM, marg=False)
par.plotContours(invFishers=None, lim=4., colors=None, fisherNames=None, path=fish.figurePath+"/planck_prior_lcdm.pdf")
# LCDMWW0
par = cosmoPar.extractParams(cosmoPar.ILCDMW0, marg=False)
par.plotContours(invFishers=None, lim=4., colors=None, fisherNames=None, path=fish.figurePath+"/planck_prior_lcdmw0.pdf")
# LCDMWW0Wa
par = cosmoPar.extractParams(cosmoPar.ILCDMW0Wa, marg=False)
par.plotContours(invFishers=None, lim=4., colors=None, fisherNames=None, path=fish.figurePath+"/planck_prior_lcdmw0wa.pdf")
# LCDMMnu
par = cosmoPar.extractParams(cosmoPar.ILCDMMnu, marg=False)
par.plotContours(invFishers=None, lim=4., colors=None, fisherNames=None, path=fish.figurePath+"/planck_prior_lcdmmnu.pdf")
# LCDMCurv
par = cosmoPar.extractParams(cosmoPar.ILCDMCurv, marg=False)
par.plotContours(invFishers=None, lim=4., colors=None, fisherNames=None, path=fish.figurePath+"/planck_prior_lcdmcurv.pdf")


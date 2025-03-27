import numpy as np
import scipy.optimize as opt
import time
import helper
import calcMuTau as calc
import warnings
warnings.filterwarnings("ignore")

'''
Main file for evaluation of SN limits on L_mu-tau models, including 4 methods:
- checkModel_FS: Compute free-streaming luminosity for a given model specified
  by (mZp, mChi/mZp, gL, gChi/gL) and using SN simulation points in the range
  rangeSim, where 62 is the point with the largest contribution to the integral.
  The parameter nSim controls, which of the SN simulations is used, where nSim=1
  is the coldest and nSim=2 is the hottest.
- checkModel_TR: Compute opacity for a given model specified by the same
  parameters as in checkModel_FS and using SN simulation points in the range
  [iSphere, iSphere+nPointsSim], where the index of the chi sphere iSphere that
  is needed to satisfy the Raffelt bound can be calculated independent of the
  specific processes. Note that it is more efficient to rule out models already
  at the level of the opacity (and not at the level of the trapping luminosity),
  because the step from the opacity to the trapping luminosity depends only on
  the kinematics of the supernova (R, T) and not on the processes in question.
  Further, the parameter approx specifies the approximation used in the calculation
  with possible values approx=exact (takes lots of time), approx=inv (fast and
  decent approximation).
  Finally, the parameter scat specifies which processes were included, with possible values
  scat=1 (only chi chi -> mu mu), scat=2 (add chi mu -> chi mu) and
  scat=4 (add chi chi -> chi chi in both s and t channel).
- getCoupling_FS: Uses checkModel_FS to find the value of muon coupling gL where
  the free-streaming luminosity satisfies the Raffelt bound. Have to specify
  a range [guessLower, guessUpper] of values for gL to check.
- getCoupling_TR: Similar to getCoupling_FS, but for trapping using checkModel_TR.

Quick explanation for parameters
- iCompton: Switch to decide whether Compton is included (1) or not (0)
- scat: Processes to be included in the calculation of the mean free path.
  Possible choices are (see 2307.03143 for more information)
  1: only annihilation
  2: annihilation + scattering
  4: like 2, but also with DM self-interactions
- approx: Approximations for the mean free path calculation. Valid values are
  "exact": Really calculate <MFP>, meaning one first has to calculate Gamma (2 numerical
  integrals) and then integrate over its inverse (one more numerical integral)
  "inv": Estimate <MFP> by <MFP^-1>^-1, this only requires one triple integral which is much easier

- nSim: Simulation to be used, see helper. Available are 0 (SFHo-s18.6 = medium T),
  1 (SFHo-s18.80 = coolest), 2 (SFHo-s20.0 = hottest), LS220 (different equation of state)  
- rangeSim: SN radius setpoint region in simulations to be used for estimating the free-streaming luminosity.
  We use rangeSim=[40,100] for the paper, but a good estimate can already be obtained with
  rangeSim=[60,65]. The largest contribution comes from the setpoint 62.
- nPointsSim: Additional SN radius setpoints to be used for estimating the mean free path (MFP)
  in the trapping regime. We start at the setpoint corresponding to the chi sphere and count from there.
  We use nPointsSim=50 for the paper, but a good estimate can already be obtained with nPointSim=10.
- guessLower, guessUpper, defaultVal: The functions getCoupling_FS, getCoupling_TR use a root finder that
  repeatedly calls checkModel_FS, checkModel_TR until it finds a suitable root. As input it requires
  the boundaries of the interval [guessLower, guessUpper]. If no solution is found, then it returns
  defaultVal.
'''

def checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim=[60,65], nSim=1, iCompton=1,
                  out=True):
    '''
    Calculate free-streaming luminosity Q for the parameter point (mZp, mChi/mZp, gL, gChi/gL)
    using parameters described above. The result is interpreted and compared with the Raffelt
    bound from the neutrino luminosity.
    '''
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    n1, n2 = rangeSim
    N = n2-n1
    dQdR = np.zeros(N)
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    t00 = time.time()
    for i in range(N):
        t0 = time.time()
        dQdR[i] = calc.dQdR(helper.mmu, mChi, mu_mu[n1+i], mu_numu[n1+i], T[n1+i], R[n1+i], mZp=mZp,
                            gChi=gChi, gL=gL, iCompton=iCompton, oneFermion=False, limit="full")
        t1 = time.time()
        if i==0 and out:
            print(f"Estimate: {(t1-t0)*N:.2f} s = {(t1-t0)/60*N:.2f} min")
    t01 = time.time()
    if out:
        print(f"Total time used: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min")
    Q = np.trapz(dQdR, x=R[n1:n2])

    if(out):
        if Q > helper.getQbound(nSim):
            print(f"### Model excluded by free-streaming ###")
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is LARGER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.getQbound(nSim)))
        else:
            print(f"### Model not excluded by free-streaming ###")
            print("Free-streaming luminosity is {0:.2e} MeV^2, this is SMALLER than the Raffelt bound {1:.2e} MeV^2.".format(Q, helper.getQbound(nSim)))
        print("NOTE: The contributions to the luminosity where calculated using only a limited number of points in the SN simulation, more explicitly points in the range rangeSim=[{0}, {1}].".format(n1, n2))
        print("Using more points INCREASES the free-streaming luminosity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return Q

def checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim=5, nSim=1, scat=2, approx="inv", iSphere=None,
                  iCompton=1, out=True):
    '''
    Calculate opacity (trapping regime) for the parameter point (mZp, mChi/mZp, gL, gChi/gL)
    using parameters described above. We calculate the opacity using the radius of the chi sphere
    that would translate into a blackbody radiation with luminosity that saturates the Raffelt bound.
    Finally, we compare the result to the value 2/3 that actually defines the location of the chi sphere,
    and use this result to infer whether the effective chi sphere of this parameter point has a
    luminosity smaller or larger than the Raffelt bound.
    Note that we could also compare results at the level of the luminosity (instead of the level of the
    opacity as we do), but this requires an extra computational effort.
    '''
    mChi = mChiOvermZp*mZp
    gChi = gL*gChiOvergL
    
    R, T, _, mu_mu, _, _, _, mu_numu = helper.unpack(nSim)
    if iSphere is None:
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim, out=False)
        if(iSphere is None):
            print(f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV")
            return None
    lambdaInv = np.zeros(nPointsSim)
    t00 = time.time()
    for i in range(nPointsSim):
        t0 = time.time()
        lambdaInv[i] = calc.lambdaInvMean(helper.mmu, mChi, mu_mu[iSphere+i], mu_numu[iSphere+i],
                                          T[iSphere+i], scat=scat, iCompton=iCompton, mZp=mZp,
                                          gChi=gChi, gL=gL, approx=approx, oneFermion=False, limit="full")
        t1 = time.time()
        if i==0 and out:
            print(f"Estimate: {(t1-t0)*nPointsSim:.2f} s = {(t1-t0)/60*nPointsSim:.2f} min")  
    t01 = time.time()
    if out:
        print(f"Total time used: {t01-t00:.2f} s = {(t01-t00)/60:.2f} min")          
    opacity = np.trapz(lambdaInv, R[iSphere:iSphere+nPointsSim])

    if out:
        if opacity < 2/3:
            print(f"### Model excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {opacity:.2e}.")
            print("This is SMALLER than the value 2/3 that defines the critical radius, resulting in a SMALLER radius of the DM sphere and therefore a Boltzmann luminosity LARGER than the Raffelt bound")
        else:
            print(f"### Model not excluded by Boltzmann trapping ({approx} approximation) ###\nOpacity at the critical radius where the corresponding Boltzmann luminosity satisfies the Raffelt bound is {opacity:.2e}.")
            print("This is LARGER than the value 2/3 that defines the critical radius, resulting in a LARGER radius of the DM sphere and therefore a Boltzmann luminosity SMALLER than the Raffelt bound")
        print("NOTE: The contributions to the opacity where calculated using only a limited number of points in the SN simulation, more explicitly the first nSimPoints={0} points after the critical point {1} where the chi sphere has to end to satisfy the Raffelt bound.".format(nPointsSim, iSphere))
        print("Using more points INCREASES the opacity, moving the couplings where the Raffelt bound is met to SMALLER values.")
    return opacity

def getCoupling_FS(mZp, mChiOvermZp, gChiOvergL, rangeSim=[60,65], nSim=1, guessLower=1e-10, guessUpper=1e0,
                   defaultVal=1, iCompton=1, outCheck=True, out=True):
    '''
    Call checkModel_FS iteratively to find the coupling gL that saturates the Raffelt bound
    for given (mZp, mChi/mZp, gChi/gL). 
    '''
    def checkFS(gL):
        Q = checkModel_FS(mZp, mChiOvermZp, gL, gChiOvergL, rangeSim, nSim, out=False)
        Qbound = helper.getQbound(nSim)
        if(outCheck):
            print("gL = {0:.2e} \t checkFS (gL) = {1:.2e}".format(gL, (Q - Qbound)/(Q + Qbound)))
        return (Q - Qbound)/(Q + Qbound)
    try:
        sol = opt.root_scalar(checkFS, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748")
    except ValueError as e:
        if len(e.args)>0 and e.args[0].__contains__("a, b must bracket a root"): #catch "no solution error"
            print(f"ERROR: No solution in the specified range [{guessLower}, {guessUpper}]. Please adapt the range.")
            return defaultVal
        else:
            raise e
    gL = sol.root
    if out:
        print("Coupling with FSLumi = RaffeltLumi: gL = {0:.1e}".format(gL))
    return gL

def getCoupling_TR(mZp, mChiOvermZp, gChiOvergL, nPointsSim=5, nSim=1, guessLower=1e-10, guessUpper=1e0,
                   defaultVal=1, scat=2, approx="inv", iCompton=1, outCheck=True, out=True):   
    '''
    Call checkModel_FS iteratively to find the coupling gL that saturates the Raffelt bound
    for given (mZp, mChi/mZp, gChi/gL). 
    '''
    # calculate iSphere
    R, T, _, _, _, _, _, _ = helper.unpack(nSim)
    mChi = mChiOvermZp*mZp
    for i in range(nPointsSim):
        iSphere = helper.getRadiusSphere(mChi, R, T, nSim, out=False) 
    if(iSphere is None):
        print(f"No radius with sufficiently large Boltzmann luminosity for mChi={mChi:.2e} MeV")
        return None
    def checkTR(gL):
        opacity = checkModel_TR(mZp, mChiOvermZp, gL, gChiOvergL, nPointsSim, nSim, scat=scat, approx=approx, iSphere=iSphere, out=False)
        if(outCheck):
            print("gL = {0:.2e} \t checkTR (gL) = {1:.2e}".format(gL, (opacity - 2/3)/(opacity + 2/3)))
        return (opacity - 2/3)/(opacity + 2/3)
    try:
        sol = opt.root_scalar(checkTR, rtol=1e-1, bracket=[guessLower, guessUpper], method="toms748")
    except ValueError as e:
        if len(e.args)>0 and e.args[0].__contains__("a, b must bracket a root"): #catch "no solution error"
            print(f"ERROR: No solution in the specified range [{guessLower}, {guessUpper}]. Please adapt the range.")
            return defaultVal
        else:
            raise e
    gL = sol.root
    if out:
        print("Coupling with opacity = 2/3: gL = {0:.1e} ({1})".format(gL, approx))
    return gL


# Examples #

checkModel_FS(3., 1/3, 1e-9, 1.)
checkModel_TR(3., 1/3, 1e-6, 1., scat=2, approx="inv")
checkModel_TR(3., 1/3, 1e-6, 1., scat=2, approx="exact")

getCoupling_FS(3., 1/3, 1)
getCoupling_TR(3., 1/3, 1, scat=2, approx="inv")
getCoupling_TR(3., 1/3, 1, scat=2, approx="exact")

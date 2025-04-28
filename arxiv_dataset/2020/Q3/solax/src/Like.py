#================================Like.py=======================================#
# Written by C. O'Hare 2020
# Contains:
# 1. InterpExpectedEvents: converts tabulated data (from AxionFuncs.py)
#    to data for arbitrary mass
# 2. ConstantObsNumberLine: contour of constant N_exp
# 3. Mass discovery limit (simple analytic formula)
# 4. Bfield discovery limit (simple analytic formula)
# 5. Background events
#==============================================================================#

from numpy import pi, sqrt, exp, zeros, size, shape, sinc, linspace, logspace
from numpy import log10, floor, log, interp, sort
from numpy import append, flipud, argmin, ones, array, vstack
from scipy.integrate import cumtrapz, quad
#from iminuit import minimize # <--- uncomment this to use minuit
from scipy.optimize import minimize # <--- comment this to use minuit
from numpy.random import poisson
from scipy.special import gammaln

#==============================================================================#
# that interpolates N_exp events at an arbitrary mass (m)
# using the tabulated data stored in R1_tab, then rescales by g
def InterpExpectedEvents(g,m,m_vals,R1_tab):
    nm = size(m_vals)
    m1 = m_vals[0]
    m2 = m_vals[-1]
    i1 = int(floor((log10(m)-log10(m_vals[0]))*(nm-1.0)/\
                    (log10(m_vals[-1])-log10(m_vals[0]))+1.0))-1
    i2 = i1+1
    if (i1<0) or (i2<0):
        i1 = 0
        i2 = 1
    N_exp = 1e40*(g**4.0)*((R1_tab[:,i1]*(m_vals[i2]-m)+\
        R1_tab[:,i2]*(m - m_vals[i1]))/(m_vals[i2]- m_vals[i1]))
    return N_exp
#==============================================================================#



#==============================================================================#
# Line of values of g that give a constant number of events N_ob
def ConstantObsNumberLine(N_ob,mi,m_vals,R1_tab):
    ni = size(mi)
    g = zeros(shape=ni)
    for i in range(0,ni):
        N_exp_10 = InterpExpectedEvents(1e-10,mi[i],m_vals,R1_tab)
        g[i] = 1e-10*(N_ob/sum(N_exp_10))**0.25
    return g
#==============================================================================#



#==============================================================================#
# Simple analytic calculation of mass discovery limit (see Dafni et al. 2020)
def MassDiscoveryLimit_Simple(m_vals,R1_tab,R0,m_DL_vals):
        nm = size(m_vals)
        n_DL = size(m_DL_vals)
        DL = zeros(shape=n_DL)
        for im in range(0,n_DL):
            m0 = m_DL_vals[im]
            i0 = int(floor((log10(m0)-log10(m_vals[0]))*(nm-1.0)\
                        /(log10(m_vals[-1])-log10(m_vals[0]))+1.0))-1
            i0 = max(i0,0)
            N = R1_tab[:,i0]
            N0 = R0
            D = sum(R1_tab[:,i0])/sum(R0)
            DL[im] = 1e-10*(9.0/sum(2*N*log(N/(D*N0))))**0.25
        return DL
#==============================================================================#




#==============================================================================#
# Simple analytic calculation of mass discovery limit
def BfieldDiscoveryLimit_Simple(m_vals,m_DL_vals,R1_tab,R0_tab):
        nm = size(m_vals)
        n_DL = size(m_DL_vals)
        DL = zeros(shape=n_DL)
        for im in range(0,n_DL):
            m0 = m_DL_vals[im]
            i0 = int(floor((log10(m0)-log10(m_vals[0]))*(nm-1.0)/\
                        (log10(m_vals[-1])-log10(m_vals[0]))+1.0))-1
            i0 = max(i0,0)
            N = R1_tab[:,i0] # Non zero B-field
            N0 = R0_tab[:,i0] # zero B-field
            D = sum(N)/sum(N0)
            DL[im] = 1e-10*(9.0/sum(2*N*log(N/(D*N0))))**0.25
        return DL
#==============================================================================#



#==============================================================================#
def BinnedBackgroundEvents(Ei,E_bins,background_level=1.0e-8,\
                            A_detector=8*0.15,Exposure=3.0):
    nE_bins = shape(E_bins)[0]
    nfine = int(shape(Ei)[0]/nE_bins)
    dN_B = A_detector*background_level*Exposure*3600*24*365*\
                        ones(shape=nE_bins*nfine)
    Background = zeros(shape=nE_bins)
    for i in range(0,nE_bins):
        Ebin = Ei[i*nfine:(i+1)*nfine]
        dNbin = dN_B[i*nfine:(i+1)*nfine]
        Background[i] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))
    return Background
#==============================================================================#






#==============================================================================#
# Useful likelihoods
# Log of poisson pdf
def lnPF(Nob,Nex):
    return sum(Nob*log(Nex) - Nex)# - gammaln(Nob+1.0)) #factorial removed for speed
def lnGF(x,mu,sig): # SUM OF LOG(GAUSSIAN PDF)
    L = (-1.0*log(sig)-0.5*log(2.0*pi)-(x-mu)**2.0/(2.0*sig**2.0))
    return L
#==============================================================================#




#==============================================================================#
# Simple 2D profile likelihood tests
# g = axion photon coupling
# Signal_10 = Signal table (from AxionFuncs.py) for gag=1e-10
# Background = Background table

# Full monte carlo for N_expts toy experiments:
def ProfileLikelihoodTest_2D_MonteCarlo(g,Signal_10,Background,n_expts=100):
    mu_true10 = sum(Signal_10)
    mu_true = mu_true10*(g/1e-10)**4.0
    f_mu = Signal_10/sum(Signal_10)
    b_true = sum(Background)
    f_b = Background/sum(Background)
    T1 = zeros(shape=n_expts)
    bnds1 = ((0, None), (0, None))
    bnds0 = ((0, None),)
    for expt in range(0,n_expts):
        N_obs = poisson(lam=(mu_true*f_mu+b_true*f_b))
        logL1 = lambda x : -1.0*lnPF(N_obs,x[0]*f_mu+x[1]*f_b)
        logL0 = lambda x : -1.0*lnPF(N_obs,0.0*f_mu+x[0]*f_b)
        res1 = minimize(logL1, array([mu_true,b_true]),bounds=bnds1)
        res0 = minimize(logL0, array([b_true]),bounds=bnds0)
        T1[expt] = -2*(logL1(res1.x)-logL0(res0.x))
    T1[T1<0.0] = 0.0
    return T1

# Asimov approximation:
def ProfileLikelihoodTest_2D_Asimov(g,Signal_10,Background):
    mu_true10 = sum(Signal_10)
    mu_true = mu_true10*(g/1e-10)**4.0
    f_mu = Signal_10/sum(Signal_10)
    b_true = sum(Background)
    f_b = Background/sum(Background)
    N_asimov = mu_true*f_mu+b_true*f_b
    bnds1 = ((0, None), (0, None))
    bnds0 = ((0, None),)
    logL1 = lambda x : -1.0*lnPF(N_asimov,x[0]*f_mu+x[1]*f_b)
    logL0 = lambda x : -1.0*lnPF(N_asimov,0.0*f_mu+x[0]*f_b)
    res1 = minimize(logL1, array([mu_true,b_true]),bounds=bnds1)
    res0 = minimize(logL0, array([b_true]),bounds=bnds0)
    T1_asimov = -2*(logL1(res1.x)-logL0(res0.x))
    return T1_asimov
#==============================================================================#






#==============================================================================#
# Sensitiviy calculations based on profile likelihood ratio test
# m_vals = values of mass used to make the table Signal_10
# m_DL = values of mass to evaluate the discovery limit at
# ng = values of g to scan over for each mass
# Nmin/max = min/max values of N_exp used to set the range of g to search over

# Asimov approximation
def ProfileLikelihood_Sensitivity(m_DL_vals,m_vals,Signal_10,Background,\
                                    Nmin=0.0,Nmax=50.0,ng=50):
    n_DL = shape(m_DL_vals)[0]
    DL = zeros(shape=n_DL)
    DL_1sig_upper = zeros(shape=n_DL)
    DL_1sig_lower = zeros(shape=n_DL)
    DL_2sig_upper = zeros(shape=n_DL)
    DL_2sig_lower = zeros(shape=n_DL)
    for i in range(0,n_DL):
        Signal_10_i = InterpExpectedEvents(1e-10,m_DL_vals[i],m_vals,Signal_10)
        N10 = sum(Signal_10_i)
        gvals = 1e-10*(linspace(Nmin,Nmax,ng)/N10)**(1.0/4.0)
        T1_asimov = zeros(shape=ng)
        for j in range(0,ng):
            T1_asimov[j] = sqrt(ProfileLikelihoodTest_2D_Asimov(gvals[j],\
                                    Signal_10_i,Background))

            if (T1_asimov[j]-1.8)>1.64:
                break
        DL_1sig_upper[i] = interp(1.64,T1_asimov[1:j]+1,gvals[1:j]) # 68% upper
        DL_1sig_lower[i] = interp(1.64,T1_asimov[1:j]-1,gvals[1:j]) # 68% lower
        DL_2sig_upper[i] = interp(1.64,T1_asimov[1:j]+1.8,gvals[1:j]) #95% upper
        DL_2sig_lower[i] = interp(1.64,T1_asimov[1:j]-1.8,gvals[1:j]) #96% lower
        DL[i] = interp(1.64,T1_asimov[1:j],gvals[1:j])
    return vstack((DL,DL_1sig_lower,DL_1sig_upper,DL_2sig_lower,DL_2sig_upper))

# Full Monte Carlo
def ProfileLikelihood_Sensitivity_MonteCarlo(m_DL_vals,m_vals,Signal_10,\
                                Background,n_expts=100,Nmin=3.0,Nmax=50.0,ng=20,
                                verbose=True):
    n_DL = shape(m_DL_vals)[0]
    DL = zeros(shape=n_DL)
    for i in range(0,n_DL):
        Signal_10_i = InterpExpectedEvents(1e-10,m_DL_vals[i],m_vals,Signal_10)
        N10 = sum(Signal_10_i)
        gvals = 1e-10*(linspace(Nmin,Nmax,ng)/N10)**(1.0/4.0)
        T1 = zeros(shape=ng)
        for j in range(0,ng):
            T = ProfileLikelihoodTest_2D_MonteCarlo(gvals[j],\
                            Signal_10_i,Background,n_expts=n_expts)
            T1[j] = sqrt(sort(T)[int(0.50*n_expts)])
            T1_asimov = sqrt(ProfileLikelihoodTest_2D_Asimov(gvals[j],\
                                        Signal_10_i,Background))

            #print(j,T1[j],T1_asimov)
            if (T1[j])>1.64:
                break
        DL[i] = interp(1.64,T1[1:j+1],gvals[1:j+1])
        if verbose:
            print(i,DL[i])
    return DL
#==============================================================================#





#==============================================================================#
# Similar to above but for the (g,m,B) likelihood function:
def ProfileLikelihoodTest_3D_Asimov(g,Signal_10,Signal_10_B,Background):
    mu_true10 = sum(Signal_10)
    mu_true10_1 = sum(Signal_10_B)
    mu_true = mu_true10*(g/1e-10)**4.0
    mu_true_1 = mu_true10_1*(g/1e-10)**4.0

    f_mu = Signal_10/sum(Signal_10)
    f_mu_1 = Signal_10_B/sum(Signal_10_B)

    b_true = sum(Background)
    f_b = Background/sum(Background)

    N_asimov = mu_true_1*f_mu_1+mu_true*f_mu+b_true*f_b
    bnds1 = ((0, None), (0, None), (0, None))
    bnds0 = ((0, None), (0, None))
    logL1 = lambda x : -1.0*lnPF(N_asimov,x[0]*f_mu_1+x[1]*f_mu+x[2]*f_b)
    logL0 = lambda x : -1.0*lnPF(N_asimov,0.0*f_mu_1+x[0]*f_mu+x[1]*f_b)
    res1 = minimize(logL1, array([mu_true_1,mu_true,b_true]),bounds=bnds1)
    res0 = minimize(logL0, array([mu_true,b_true]),bounds=bnds0)
    T1_asimov = -2*(logL1(res1.x)-logL0(res0.x))
    return T1_asimov

# Signal_10 -> B=0 photon number table for gag=1e-10
# Signal_10_B -> B!=0 photon number table for gag=1e-10
def ProfileLikelihood_BfieldSensitivity(m_DL_vals,m_vals,Signal_10,Signal_10_B,\
                        Background,Nmin=0.0,Nmax=10.0,ng=50):
    n_DL = shape(m_DL_vals)[0]
    DL = zeros(shape=n_DL)
    DL_1sig_upper = zeros(shape=n_DL)
    DL_1sig_lower = zeros(shape=n_DL)
    DL_2sig_upper = zeros(shape=n_DL)
    DL_2sig_lower = zeros(shape=n_DL)
    for i in range(0,n_DL):
        Signal_10_i = InterpExpectedEvents(1e-10,m_DL_vals[i],m_vals,Signal_10)
        Signal_10_i_1 = InterpExpectedEvents(1e-10,m_DL_vals[i],m_vals,Signal_10_B)

        N10 = sum(Signal_10_i_1)
        gvals = 1e-10*(linspace(Nmin,Nmax,ng)/N10)**(1.0/4.0)
        T1_asimov = zeros(shape=ng)
        for j in range(0,ng):
            T1_asimov[j] = sqrt(ProfileLikelihoodTest_3D_Asimov(gvals[j],\
                                    Signal_10_i,Signal_10_i_1,Background))

            if (T1_asimov[j]-1.8)>1.64:
                break
        DL_1sig_upper[i] = interp(1.64,T1_asimov[1:j]+1,gvals[1:j]) # 68%
        DL_1sig_lower[i] = interp(1.64,T1_asimov[1:j]-1,gvals[1:j]) # 68%
        DL_2sig_upper[i] = interp(1.64,T1_asimov[1:j]+1.8,gvals[1:j]) # 95%
        DL_2sig_lower[i] = interp(1.64,T1_asimov[1:j]-1.8,gvals[1:j]) # 95%
        DL[i] = interp(1.64,T1_asimov[1:j],gvals[1:j])
    return vstack((DL,DL_1sig_lower,DL_1sig_upper,DL_2sig_lower,DL_2sig_upper))
#==============================================================================#



















#==============================================================================#
#==============================================================================#
#==============================================================================#
#==============================================================================#
#==============================================================================#
# OLD CODE:
#==============================================================================#
#==============================================================================#
#==============================================================================#
#==============================================================================#



#==============================================================================#
# Old likelihood analysis:

# 2D likelihood for (g,m)
def llhood2(X,N_obs,m_vals,R1_tab):
    m = X[1]
    g = 10.0**X[0]
    N_exp = InterpExpectedEvents(g,m,m_vals,R1_tab)
    LL = -1.0*lnPF(N_obs,N_exp)
    return LL

# Profile likelihood for (g,m=0)
def llhood1(X,N_obs,R0):
    g = 10.0**X[0]
    N_exp = 1e40*(g**4.0)*R0
    LL = -1.0*lnPF(N_obs,N_exp)
    return LL

# Profile likelihood for (m)
def llhood2_marg(m,N_obs,m_vals,R1_tab):
    N_exp_10 = InterpExpectedEvents(1e-10,m,m_vals,R1_tab)
    g0 = ((sum(N_obs)/sum(N_exp_10))**0.25)*1e-10
    LL = llhood2([log10(g0),m],N_obs,m_vals,R1_tab)
    return LL

# Profile likelihood for (m=0)
def llhood2_marg0(N_obs,R0):
    N_exp = (sum(N_obs)/sum(R0))*R0
    LL = -1.0*lnPF(N_obs,N_exp)
    return LL

# Simple analytic calculation of mass discovery limit for (m,g) likelihood
def MassDiscoveryLimit_Simple(m_vals,R1_tab,R0,m_DL_vals):
        nm = size(m_vals)
        n_DL = size(m_DL_vals)
        DL = zeros(shape=n_DL)
        for im in range(0,n_DL):
            m0 = m_DL_vals[im]
            i0 = int(floor((log10(m0)-log10(m_vals[0]))*(nm-1.0)\
                            /(log10(m_vals[-1])-log10(m_vals[0]))+1.0))-1
            N = R1_tab[:,i0]
            N0 = R0
            D = sum(R1_tab[:,i0])/sum(R0)
            DL[im] = 1e-10*(9.0/sum(2*N*log(N/(D*N0))))**0.25
        return DL
#==============================================================================#



# Expanded Likelihoods and Discovery limit using Minuit:

# 3D likelihood for (g,m,dPhi)
def fullhood2(X,N_obs,m_vals,R1_tab,Phi0,dPhi0):
    dPhi = X[2]
    m = X[1]
    g = 10.0**X[0]
    N_exp = InterpExpectedEvents(g,m,m_vals,R1_tab)*(1+dPhi/Phi0)
    LL = -1.0*lnPF(N_obs,N_exp) - lnGF(Phi0+dPhi,Phi0,dPhi0)
    return LL

# Profile likelihood for (g,m=0,dPhi)
def fullhood1(X,N_obs,R0,Phi0,dPhi0):
    dPhi = X[1]
    g = 10.0**X[0]
    N_exp = 1e40*(g**4.0)*R0*(1+dPhi/Phi0)
    LL = -1.0*lnPF(N_obs,N_exp) - lnGF(Phi0+dPhi,Phi0,dPhi0)
    return LL

def MassDiscoveryLimit_Minuit(m_vals,R1_tab,R0,Phi0,dPhi0,m_DL_vals,gmin=1e-12,gmax=1e-7,ng=100):
    nm = size(m_vals)
    n_DL = size(m_DL_vals)
    DL = zeros(shape=(n_DL))
    g_vals = logspace(log10(gmin),log10(gmax),ng)
    for im in range(0,n_DL):
        for j in range(0,ng):
            g = g_vals[j]
            m0 = m_DL_vals[im]
            N_obs = InterpExpectedEvents(g,m0,m_vals,R1_tab)

            #print log10(g),log10(m0),sum(N_obs)
            D12_prev = 0.0
            g_prev = g
            if sum(N_obs)>3:
                # ----- Massive case -------- #
                L2 = -1.0*lnPF(N_obs,N_obs) - 1.0*lnGF(Phi0,Phi0,dPhi0)

                #------ Massless case ------#
                X_in1 = append(log10(g),dPhi0/10.0)
                res = minimize(fullhood1, X_in1, args=(N_obs,R0,Phi0,dPhi0))
                L1 = res.fun

                # Test statistic
                D12 = -2.0*(L2-L1) # significance for measuring mass
                #print m0,g,L2,fullhood1(X_in1,N_obs,R0,Phi0,dPhi0),D12
                if D12>9.0: # Median 3sigma detection -> D = 9
                    DL[im] = 10.0**(interp(9.0,[D12_prev,D12],[log10(g_prev),log10(g)]))
                    break
                g_prev = g # Reset for interpolation
                D12_prev = D12
    return DL
#==============================================================================#


#==============================================================================#
# Function for calculating the mass estimation discovery limit
# (Fig.6 of the paper)
# Can work in two modes:
#
# Mode 1 when gmin_vals = scalar: The scan over coupling values begins with
# the value that produced N_obs=2 events
#
# Mode 2 (when gmin_vals = array of size (n_DL)): The scan begins at the values
# specified
#
# The two modes are needed because the first scan over the parameter space is needed
# to eliminate all of the spurious regions at low and high masses that give rise
# to likelihoods within the confidence interval band
# (i.e. likelihood ratio>-4 for example) but are no where near the correct mass
# (i.e. within err away from m0)
def MassEstimationDiscoveryLimit(err,m_vals,R0,R1_tab,m_DL_vals,sigmas=2,gmin_vals=1.0,gmax=1e-9,ng=500,nL=1000):
    itot = nL
    nm = size(m_vals)
    n_DL = size(m_DL_vals)
    DLmerr = zeros(shape=(n_DL))
    for im in range(0,n_DL):
        m0 = m_DL_vals[im]
        i0 = int(floor((log10(m0)-log10(m_vals[0]))\
                       *(nm-1.0)/(log10(m_vals[-1])-log10(m_vals[0]))+1.0))-1
        iupper = int(itot*i0/nm)
        ilower = itot-iupper
        mlower = flipud(logspace(log10(m_vals[1]),log10(m0*(1-err/2)),ilower))
        mupper = logspace(log10(m0*(1+err/2)),log10(m_vals[-2]),iupper)
        if size(gmin_vals)>1:
            gmin = gmin_vals[im]
        else:
            N = R1_tab[:,i0]
            N0 = R0
            D = sum(R1_tab[:,i0])/sum(R0)
            gmin = 1e-10*(1.0/sum(2*N*log(N/(D*N0))))**0.25

        g_vals = logspace(log10(gmin),log10(gmax),ng)
        for j in range(0,ng):
            g0 = g_vals[j]
            N_obs = R1_tab[:,i0]*(g0/1e-10)**4.0
            Lmax = llhood2_marg(m0,N_obs,m_vals,R1_tab)
            Lprof0 = -2*(llhood2_marg0(N_obs,R0) - Lmax)
            Lprofend = -2*(llhood2_marg(m_vals[-2],N_obs,m_vals,R1_tab) - Lmax)
            if (Lprof0<-sigmas**2.0)&(Lprofend<-sigmas**2.0):
                c = True
                for ii in range(0,ilower):
                    Lprof = -2*(llhood2_marg(mlower[ii],N_obs,m_vals,R1_tab) - Lmax)
                    if Lprof>-sigmas**2.0:
                        c = False
                        break
                for ii in range(0,iupper):
                    Lprof = -2*(llhood2_marg(mupper[ii],N_obs,m_vals,R1_tab) - Lmax)
                    if Lprof>-sigmas**2.0:
                        c = False
                        break
                if c:
                    DLmerr[im] = g0
                    break

    return DLmerr
#==============================================================================#




#==============================================================================#
# Given an input mass and coupling (m0,g0), calculates the confidence interval
# around the best fit mass to a given significance level (sigmas)
# also requires err_l and err_u which is roughly the fractional size over which
# to search (needs to be very large ~100% for most values of m0)
# This function currently not quite extensive enough to give accurate results
# Unless one requires err<0.01
def MassMeasurement(m0,g0,err_l,err_u,sigmas,m_vals,R1_tab,nL=100):
    nm = size(m_vals)
    mlim = []
    N_obs = InterpExpectedEvents(g0,m0,m_vals,R1_tab)
    mi = linspace((1-err_l)*m0,(1+err_u)*m0,nL)
    LL = zeros(shape=nL)
    maxL = llhood2([log10(g0),m0],N_obs,m_vals,R1_tab)
    up = True
    for ii in range(0,nL):
        m = mi[ii]
        dL = -2*(llhood2_marg(m,N_obs,m_vals,R1_tab)-maxL)
        if up:
            if dL>-sigmas**2.0:
                mlim = append(mlim,m)
                up = False
        else:
            if dL<-sigmas**2.0:
                mlim = append(mlim,m)
                up = True
    return mlim
#==============================================================================#


#==============================================================================#
# Old way of calculating the Mass estimate discovery limit, currently it fails
# for most axion masses because the likelihood is so badly behaved
def MassErrorDiscoveryLimit_Old(err,m_vals,R0,R1_tab,m_DL_vals,gmin=1e-12,gmax=1e-7,ng=100,nL=100):
    nLike = nL
    nm = size(m_vals)
    n_DL = size(m_DL_vals)
    DLmerr = zeros(shape=(n_DL))
    g_vals = logspace(log10(gmin),log10(gmax),ng)
    for im in range(0,n_DL):
        m0 = m_DL_vals[im]
        for j in range(0,ng):
            g0 = g_vals[j]
            N_obs = InterpExpectedEvents(g0,m0,m_vals,R1_tab)
            Lmax = llhood2_marg(m0,N_obs,m_vals,R1_tab)
            D0 = -2*(llhood2_marg0(N_obs,R0)-Lmax)
            if D0<-1:
                X_in2 = [log10(g0),m0]
                maxL = llhood2(X_in2,N_obs,m_vals,R1_tab)
                Lu = -2*(llhood2_marg(m0*(1+0.5),N_obs,m_vals,R1_tab)-maxL)
                Ll = -2*(llhood2_marg(m0*(1-0.5),N_obs,m_vals,R1_tab)-maxL)
                if (Lu<-1)&(Ll<-1):
                    mlim = MassMeasurement(m0,g0,err,err,1,m_vals,R1_tab,nL=nLike)
                    if size(mlim)<=2:
                        rel_err = (max(mlim)-min(mlim))/(2*m0)
                        #rel_err = (max(log10(mlim))-min(log10(mlim)))/(2*log10(m0))
                        if rel_err<err:
                            DLmerr[im] = g0
                            break

    return DLmerr
#==============================================================================#
